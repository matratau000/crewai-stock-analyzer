import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
    "DEFAULT_LLM": "groq",
    "DEFAULT_OLLAMA_MODEL": "gemma2:9b",
    "CHART_SAVE_PATH": "./charts/",
    "REPORT_SAVE_PATH": "./reports/"
}
