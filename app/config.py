from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    GENAI_API_KEY: str = os.getenv("GENAI_API_KEY") or ""
    GEMINI_MODEL: str

    class Config:
        env_file = ".env"
        
settings = Settings()