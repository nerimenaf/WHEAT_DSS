from dataclasses import dataclass
import os

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    openweather_api_key: str | None = os.getenv("OPENWEATHER_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openweather_forecast_url: str = "https://api.openweathermap.org/data/2.5/forecast"
    units: str = "metric"
    top_k: int = 3
    default_backend: str = "ollama"
    default_openai_model: str = "gpt-4o-mini"
    default_ollama_model: str = "mistral"
    temperature: float = 0.2

settings = Settings()