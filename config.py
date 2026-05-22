import os
from enum import Enum 
from dotenv import load_dotenv

load_dotenv()

class ModelName(Enum):
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B  = "llama-3.1-8b-instant"
    MIXTRAL_8X7B  = "mixtral-8x7b-32768"


MODEL_PRICING = {
    ModelName.LLAMA_3_3_70B: {"input": 0.59, "output": 0.79},
    ModelName.LLAMA_3_1_8B:  {"input": 0.05, "output": 0.08},
    ModelName.MIXTRAL_8X7B:  {"input": 0.24, "output": 0.24},
}

FALLBACK_CHAIN = [
    ModelName.LLAMA_3_3_70B, 
    ModelName.LLAMA_3_1_8B, 
    ModelName.MIXTRAL_8X7B
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
IS_MOCK_MODE = False