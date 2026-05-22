import json
import random
import asyncio
import hashlib
from typing import AsyncGenerator, Dict, Any, Tuple
from groq import AsyncGroq, GroqError
from dataclasses import dataclass
from config import ModelName, FALLBACK_CHAIN, GROQ_API_KEY, IS_MOCK_MODE

@dataclass
class PromptTemplate:
    name: str
    version: str
    template: str
    model: ModelName = ModelName.LLAMA_3_3_70B
    max_output_tokens: int = 1024


PROMPT_TEMPLATES = {
    "general_chat": {
        "v1": PromptTemplate(
            name="general_chat",
            version="v1",
            template=(
                "You are a helpful AI assistant. Answer the user's question clearly and concisely.\n\n"
                "User question: {query}"
            ),
        ),
        "v2": PromptTemplate(
            name="general_chat",
            version="v2",
            template=(
                "You are an AI assistant that gives precise, actionable answers. "
                "If you are unsure, say so. Never fabricate information.\n\n"
                "Question: {query}\n\nAnswer:"
            ),
        ),
    },
    "rag_answer": {
        "v1": PromptTemplate(
            name="rag_answer",
            version="v1",
            template=(
                "Answer the question using ONLY the provided context. "
                "If the context does not contain the answer, say 'I don't have enough information.'\n\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ),
            max_output_tokens=512,
        ),
    },
    "code_review": {
        "v1": PromptTemplate(
            name="code_review",
            version="v1",
            template=(
                "You are a senior software engineer performing a code review. "
                "Identify bugs, security issues, and performance problems. "
                "Be specific. Reference line numbers.\n\n"
                "Code:\n```\n{code}\n```\n\nReview:"
            ),
            model=ModelName.LLAMA_3_3_70B,
            max_output_tokens=2048,
        ),
    },
}


AB_EXPERIMENTS = {
    "general_chat_v2_test": {
        "template": "general_chat",
        "control": "v1",
        "variant": "v2",
        "traffic_pct": 10,
    },
}


def select_prompt(template_name, user_id, variables):
    versions = PROMPT_TEMPLATES.get(template_name)
    if not versions:
        raise ValueError(f"Unknown template: {template_name}")

    version = "v1"
    for exp_name, exp in AB_EXPERIMENTS.items():
        if exp["template"] == template_name:
            bucket = int(hashlib.md5(f"{user_id}:{exp_name}".encode()).hexdigest(), 16) % 100
            if bucket < exp["traffic_pct"]:
                version = exp["variant"]
            else:
                version = exp["control"]
            break

    template = versions.get(version, versions["v1"])
    rendered = template.template.format(**variables)
    return template, rendered

class LLMEngine:
    def __init__(self):
        # Instantiate the thread-safe Async Groq client
        self.client = AsyncGroq(api_key=GROQ_API_KEY)

    async def stream_chat(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        preferred_model: ModelName
    ) -> AsyncGenerator[str, None]:
        """
        Executes a streaming request against Groq. Automatically captures 429/500 
        exceptions and transparently rolls over to fallback models in the chain.
        """
        # Build local adaptive fallback loop chain matching priority settings
        chain = list(FALLBACK_CHAIN)
        if preferred_model in chain:
            chain.remove(preferred_model)
            chain.insert(0, preferred_model)

        for current_model in chain:
            try:
                if IS_MOCK_MODE:
                    yield json.dumps({
                        "type": "meta", 
                        "model": current_model.value, 
                        "input_tokens": 12, "output_tokens": 24
                    })
                    yield json.dumps({"type": "token", "text": "Mock token streaming validation..."})
                    yield json.dumps({"type": "done", "final_text": "Mock token streaming validation..."})
                    return

                # Establish streaming runtime payload connection
                stream = await self.client.chat.completions.create(
                    model=current_model.value,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    stream=True
                )

                accumulated_text = []
                async for chunk in stream:
                    # Capture real-time usage data if supplied inline by provider chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        yield json.dumps({
                            "type": "meta",
                            "model": current_model.value,
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens
                        })

                    delta = chunk.choices[0].delta.content if chunk.choices else ""
                    if delta:
                        accumulated_text.append(delta)
                        yield json.dumps({"type": "token", "text": delta})

                # Terminate stream cleanly and pass data up for cache ingestion
                yield json.dumps({
                    "type": "done", 
                    "final_text": "".join(accumulated_text)
                })
                return

            except (GroqError, Exception) as e:
                # Log or suppress failure internally to execute subsequent fallback models silently
                print(f"Warning: Model failure on {current_model.value}. Advancing fallback. Err: {e}")
                continue

        # Critical circuit break failure state
        yield json.dumps({
            "type": "error", 
            "message": "Critical: Model failover pool entirely exhausted."
        })