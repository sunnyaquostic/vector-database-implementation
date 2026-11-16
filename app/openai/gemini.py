from google import genai

class Client:
    def __init__(self, api_key: str = "api_key=YOUR_API_KEY_HERE"):
        self.client = genai.Client(api_key=api_key)
        
    def models(self):
        return self.client.models
