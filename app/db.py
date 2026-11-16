import chromadb 
from chromadb.config import Settings
from pydantic import BaseModel, Field
from typing import List, Optional
import os

class Database:
    def __init__(self, db_dir: str, name:str):
        self.conn = chromadb.PersistentClient(path=f"{os.getcwd()}/{db_dir}")
        self.collection = self.conn.get_or_create_collection(name=name)
        
    
    def add(self, *, ids: list, embeddings: list, texts: list) -> str:
        self.collection.upsert(
             ids=ids,
             documents=texts,
             embeddings=embeddings
        )
        
        return "Embeddings added to ChromaDB collection successfully."
    
    
    def read(self, query_embedding: list, no_of_result: int = 5) -> list:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=no_of_result
        )
        
        return results