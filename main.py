# main.py
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import all custom modular components
from config import ModelName
from engine import LLMEngine, select_prompt
from cache import RedisSemanticCache, CostTracker
from guardrails import validate_input, validate_output

# ============================================================================
# FastAPI Lifespan (Startup / Shutdown Management)
# ============================================================================
# Instantiate shared singleton infrastructure instances
cost_tracker = CostTracker()
llm_engine = LLMEngine()

# Instantiate Redis Semantic Cache with a distance threshold and 1-hour TTL
semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    distance_threshold=0.3,
    ttl_seconds=3600
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block executes exactly once when web application boots up
    await semantic_cache.initialize_index()
    yield
    # Clean teardown when server processes exit
    await semantic_cache.client.close()

app = FastAPI(title="Production-Grade Resilient Groq Gateway", version="2.0.0", lifespan=lifespan)

# ============================================================================
# Pydantic Request/Response Payload Validations
# ============================================================================
class ChatRequest(BaseModel):
    user_id: str = Field(..., example="usr_dev_99", description="Unique identifier for user session routing and A/B tracking.")
    template_name: str = Field(default="general_chat", example="general_chat", description="The registered layout prompt target mapping.")
    variables: dict = Field(..., example={"query": "Explain quantum computing simply"}, description="Key-value tokens injected directly into prompt layout arrays.")
    preferred_model: ModelName = Field(default=ModelName.LLAMA_3_3_70B, description="The primary requested model execution target.")

class MetricsSummary(BaseModel):
    total_requests: int
    total_cost_usd: float
    top_users: dict

# ============================================================================
# Gateway Orchestration Generator Pipeline
# ============================================================================
async def request_orchestration_pipeline(payload: ChatRequest):
    """
    Asynchronous state machine managing the lifecycle of an LLM execution flow:
    Guardrails -> Cache Interception -> Engine Execution -> Output Evaluation -> Ingestion.
    """
    user_id = payload.user_id
    variables = payload.variables
    preferred_model = payload.preferred_model
    
    # Extract the fundamental core question out of the input variables dictionary
    raw_query = variables.get("query", variables.get("code", ""))

    # 1. RUN INPUT GUARDRAILS
    if not validate_input(raw_query):
        yield json.dumps({"type": "error", "message": "Security Alert: Input violated safety guardrail policies."})
        return

    # 2. INTERCEPT VIA REDIS SEMANTIC CACHE (Now awaitable and async)
    cache_hit = await semantic_cache.get(raw_query)
    if cache_hit:
        yield json.dumps({
            "type": "meta", 
            "model": f"CACHE_HIT (Sim: {cache_hit['similarity']})", 
            "input_tokens": 0, "output_tokens": 0
        })
        yield json.dumps({"type": "token", "text": cache_hit["response"]})
        yield json.dumps({"type": "done", "final_text": cache_hit["response"]})
        return

    # 3. SELECT AND RENDER A/B PROMPT VARIANT
    try:
        prompt_template, rendered_user_prompt = select_prompt(payload.template_name, user_id, variables)
    except Exception as e:
        yield json.dumps({"type": "error", "message": f"Prompt Generation Error: {str(e)}"})
        return

    target_model = preferred_model if preferred_model else prompt_template.model
    system_instruction = "You are a secure, high-performance production enterprise AI utility."

    # 4. INITIALIZE CHAT STREAM ENGINE LOOP WITH FALLBACK RESILIENCE
    actual_model_used = target_model.value
    input_tokens_used = 0
    output_tokens_used = 0
    accumulated_response_text = ""

    async for chunk in llm_engine.stream_chat(system_instruction, rendered_user_prompt, target_model):
        data = json.loads(chunk)
        packet_type = data.get("type")

        if packet_type == "meta":
            actual_model_used = data.get("model")
            input_tokens_used = data.get("input_tokens", 0)
            output_tokens_used = data.get("output_tokens", 0)
            yield chunk  
            
        elif packet_type == "token":
            accumulated_response_text += data.get("text", "")
            yield chunk  
            
        elif packet_type == "done":
            # 5. RUN OUTPUT GUARDRAILS
            if not validate_output(accumulated_response_text):
                yield json.dumps({"type": "error", "message": "Security Alert: Generated text violated output quality policies."})
                return

            # 6. INGEST RESPONSE INTO REDIS SEMANTIC CACHE MATRIX (Now awaitable and async)
            await semantic_cache.put(raw_query, accumulated_response_text)

            # 7. COMMIT ACCOUNTING METRICS TRANSACTIONS TO TELEMETRY STORAGE
            try:
                model_enum = ModelName(actual_model_used)
                cost_tracker.record(user_id, model_enum, input_tokens_used, output_tokens_used)
            except Exception:
                cost_tracker.record(user_id, target_model, input_tokens_used, output_tokens_used)

            yield chunk  
            
        elif packet_type == "error":
            yield chunk

# ============================================================================
# API Endpoint Controller Handlers
# ============================================================================
@app.post("/v1/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    """
    Primary API gateway endpoint. Wraps our asynchronous orchestration lifecycle 
    pipeline inside a streaming text/event-stream HTTP connection block.
    """
    return StreamingResponse(
        request_orchestration_pipeline(payload), 
        media_type="text/event-stream"
    )

@app.get("/v1/telemetry/costs", response_model=MetricsSummary)
async def get_cost_metrics():
    """
    Retrieves operational cost summaries, request totals, and top resource usage aggregations.
    """
    return cost_tracker.get_summary()