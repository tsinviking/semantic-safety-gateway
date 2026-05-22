# test_pipeline.py
import time
import asyncio
import httpx
from uvicorn import Config, Server
from main import app

async def run_tests():
    # 1. Start the FastAPI server locally in the background
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="warning")
    server = Server(config)
    server_task = asyncio.create_task(server.serve())
    
    # Allow the server a moment to spin up and download/initialize the HF embedding space
    await asyncio.sleep(3)

    payload = {
        "user_id": "tester_123",
        "template_name": "general_chat",
        "variables": {"query": "How do vectors work in software engineering?"},
        "preferred_model": "llama-3.3-70b-versatile"
    }

    async with httpx.AsyncClient() as client:
        # ====================================================================
        # RUN TEST 1: The Initial Request (Cache Miss -> Engine Streaming)
        # ====================================================================
        print("\n=== RUNNING TEST 1: Initial Prompt (Expecting Cache Miss) ===")
        start_time = time.time()
        
        async with client.stream("POST", "http://127.0.0.1:8000/v1/chat/stream", json=payload) as response:
            async for chunk in response.aiter_text():
                if chunk.strip():
                    print(f"Received Chunk: {chunk.strip()}")
                    
        print(f"Test 1 Completed in: {round(time.time() - start_time, 2)} seconds")

        # ====================================================================
        # RUN TEST 2: Semantic Matching Request (Expecting Lightning Cache Hit)
        # ====================================================================
        print("\n=== RUNNING TEST 2: Altered Phrasing (Expecting Semantic Cache Hit) ===")
        # Changing the phrasing slightly to prove Hugging Face embeddings catch semantics, not exact words
        payload["variables"]["query"] = "Can you explain how vectors function inside code?"
        start_time = time.time()
        
        async with client.stream("POST", "http://127.0.0.1:8000/v1/chat/stream", json=payload) as response:
            async for chunk in response.aiter_text():
                if chunk.strip():
                    print(f"Received Chunk: {chunk.strip()}")
                    
        print(f"Test 2 (Cache Interception) Completed in: {round(time.time() - start_time, 4)} seconds")

        # ====================================================================
        # RUN TEST 3: Telemetry Review
        # ====================================================================
        print("\n=== RUNNING TEST 3: Telemetry Cost Validation ===")
        metrics_resp = await client.get("http://127.0.0.1:8000/v1/telemetry/costs")
        print(f"Metrics Output: {metrics_resp.json()}")

    # Teardown background server
    server.should_exit = True
    await server_task

if __name__ == "__main__":
    asyncio.run(run_tests())