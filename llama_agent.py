from llama_stack_client import LlamaStackClient, Agent
from typing import Optional, List, Dict

class LlamaAgent:
    def __init__(self, base_url="http://localhost:8321"):
        """Initialize Llama Stack client and agent."""
        self.client = LlamaStackClient(api_key="9b6c7caa-68ac-4f87-93e9-eed31b9a95d1")
        
        # Get available models
        try:
            models = self.client.models.list()
            self.model_id = next(m for m in models if m.model_type == "llm").identifier
            print(f"Using LLM model: {self.model_id}")
            
            # Create an agent with appropriate instructions
            self.agent = Agent(
                self.client,
                model=self.model_id,
                instructions="You are a helpful assistant that helps users find and delete PNG images from their desktop. When asked to find images, identify the relevant criteria (date, content, etc.) and search for matching images. When asked to delete images, always confirm before deletion.",
            )
            self.session_id = self.agent.create_session("png_cleanup_session")
        except Exception as e:
            print(f"Error initializing Llama agent: {str(e)}")
            self.model_id = None
            self.agent = None
            self.session_id = None
    
    def rewrite_query(self, prompt: str) -> str:
        """Rewrite user query to be more effective for image caption search."""
        if not self.client or not self.model_id:
            return prompt
            
        try:
            response = self.client.inference.chat_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites queries to be more effective for image caption search."},
                    {"role": "user", "content": f"Rephrase this as a search query for image captions: {prompt}"}
                ],
                stream=False
            )
            return response.text
        except Exception as e:
            print(f"Error rewriting query: {str(e)}")
            return prompt
    
    def understand_query(self, query: str) -> Dict:
        """Parse user query to understand intent (find or delete) and criteria."""
        if not self.agent or not self.session_id:
            # Fallback simple parsing
            intent = "delete" if "delete" in query.lower() else "find"
            return {"intent": intent, "query": query}
            
        try:
            response = self.agent.create_turn(
                messages=[{"role": "user", "content": f"Parse this query and extract the intent (find or delete) and criteria (date, content description, etc.): '{query}'"}],
                session_id=self.session_id,
            )
            
            # Try to extract structured information
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    return parsed
                except:
                    pass
            
            # Fallback to simple text analysis
            intent = "delete" if "delete" in query.lower() or "delete" in response.text.lower() else "find"
            return {"intent": intent, "query": query, "analysis": response.text}
            
        except Exception as e:
            print(f"Error understanding query: {str(e)}")
            # Fallback simple parsing
            intent = "delete" if "delete" in query.lower() else "find"
            return {"intent": intent, "query": query}
    
    def confirm_deletion(self, images: List[Dict], query: str) -> str:
        """Ask user to confirm deletion of specific images."""
        if not self.agent or not self.session_id:
            # Fallback confirmation message
            return f"Are you sure you want to delete these {len(images)} images matching '{query}'?"
            
        try:
            image_list = "\n".join([f"- {img['path']} (Caption: {img['caption']})" for img in images])
            response = self.agent.create_turn(
                messages=[{"role": "user", "content": f"Based on the query '{query}', I found these images:\n{image_list}\nShould I delete them? Generate a confirmation message for the user."}],
                session_id=self.session_id,
            )
            return response.text
        except Exception as e:
            print(f"Error generating confirmation: {str(e)}")
            # Fallback confirmation message
            return f"Are you sure you want to delete these {len(images)} images matching '{query}'?"
