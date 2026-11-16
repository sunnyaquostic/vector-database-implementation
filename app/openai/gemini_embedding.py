from google.genai import types
from .gemini import Client
from app.config import settings
from typing import List

class EmbeddingClient:
    def __init__(self, output_dimensionality: int = 768, task_type: str = "SEMANTIC_SIMILARITY"):
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self.gemini = Client(settings.GENAI_API_KEY)
        
    def create_embeddings(self, texts = list[str]):
        result = self.gemini.client.models.embed_content(
            model=settings.GEMINI_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.output_dimensionality)
        )
        
        return result
    
    def retrieve_embeddings(self, query_text: str):
        query_embedding = self.gemini.client.models.embed_content(
                                model=settings.GEMINI_MODEL,
                                contents=[query_text],
                                config=types.EmbedContentConfig(
                                    task_type=self.task_type,
                                    output_dimensionality=self.output_dimensionality   
                                )
                            ).embeddings[0].values
        
        return query_embedding