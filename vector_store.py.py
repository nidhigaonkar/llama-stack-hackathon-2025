from typing import List, Dict, Optional
import json
import re
from datetime import datetime
from llama_stack_client import LlamaStackClient, RAGDocument

class VectorStore:
    def __init__(self, llama_agent=None, vector_db_id="png_image_vector_db"):
        """Initialize the vector store with Llama Stack integration."""
        self.processed_images = []
        self.llama_agent = llama_agent
        self.vector_db_id = vector_db_id
        self.client = None
        
        if llama_agent:
            self.client = llama_agent.client
            self.setup_vector_db_with_llama_stack()
    
    def setup_vector_db_with_llama_stack(self):
        """Set up vector database with Llama Stack."""
        if not self.client:
            return
            
        # Get embedding model from available models
        embedding_models = [m for m in self.client.models.list() if m.model_type == "embedding"]
        if not embedding_models:
            print("No embedding models available. Vector store setup failed.")
            return
            
        embedding_model = embedding_models[0]
        embedding_model_id = embedding_model.identifier
        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        
        # Register vector DB with Llama Stack
        try:
            self.client.vector_dbs.register(
                vector_db_id=self.vector_db_id,
                embedding_model=embedding_model_id,
                embedding_dimension=embedding_dimension,
                provider_id="faiss",
            )
            print(f"Vector database '{self.vector_db_id}' registered successfully")
        except Exception as e:
            print(f"Error registering vector database: {str(e)}")
    
    def add_images(self, processed_images: List[Dict[str, str]]):
        """Add processed images to the local store."""
        self.processed_images.extend(processed_images)
        
        # If Llama Stack is available, add to vector store
        if self.client:
            self.add_images_to_llama_stack(processed_images)
    
    def add_images_to_llama_stack(self, processed_images: List[Dict[str, str]]):
        """Add processed images to Llama Stack vector DB."""
        if not self.client:
            return
            
        # Convert processed images to RAG documents
        documents = []
        for img in processed_images:
            document = RAGDocument(
                document_id=img['path'],
                content=f"Image Caption: {img['caption']}\nCreation Date: {img['creation_date']}\nCreation Time: {img['creation_time']}\nFile Path: {img['path']}",
                mime_type="text/plain",
                metadata={
                    "path": img['path'],
                    "creation_date": img['creation_date'],
                    "creation_time": img['creation_time']
                }
            )
            documents.append(document)
        
        # Insert documents into vector DB
        try:
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=50,
            )
            print(f"Added {len(documents)} images to Llama Stack vector database")
        except Exception as e:
            print(f"Error adding images to Llama Stack: {str(e)}")
    
    def search_images(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for images based on query using both traditional and RAG methods."""
        # If Llama Stack is available, use RAG search
        if self.client and self.llama_agent:
            return self.search_with_llama_rag(query, top_k)
        
        # Fallback to traditional search
        return self.traditional_search(query, top_k)
    
    def traditional_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search as fallback."""
        results = []
        query_lower = query.lower()
        
        # Parse date information if present
        date_range = self.parse_date_query(query)
        
        for img in self.processed_images:
            score = 0
            # Check caption for keyword match
            if query_lower in img['caption'].lower():
                score += 0.8
            
            # Check date match if date range is specified
            if date_range["start_date"] and self.is_date_in_range(img['creation_date'], date_range):
                score += 0.9
                
            if score > 0:
                result = img.copy()
                result['relevance_score'] = score
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def search_with_llama_rag(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for images using Llama Stack RAG."""
        try:
            # Rewrite query to be more effective for RAG
            rewritten_query = self.llama_agent.rewrite_query(query)
            
            # Use RAG tool to search
            response = self.client.tool_runtime.rag_tool.query(
                vector_db_ids=[self.vector_db_id],
                content=rewritten_query,
                top_k=top_k
            )
            
            # Process results
            results = []
            seen_paths = set()
            
            for chunk in response:
                # Extract path from chunk content
                path_match = re.search(r'File Path: (.*?)(?:\n|$)', chunk.content)
                if path_match:
                    path = path_match.group(1).strip()
                    
                    # Avoid duplicates
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    
                    # Find corresponding image in our local store
                    for img in self.processed_images:
                        if img['path'] == path:
                            result = img.copy()
                            result['relevance_score'] = chunk.score if hasattr(chunk, 'score') else 0.9
                            results.append(result)
                            break
            
            # Parse date information if present
            date_range = self.parse_date_query(query)
            if date_range["start_date"]:
                results = self.filter_by_date(results, date_range)
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in RAG search: {str(e)}")
            # Fallback to traditional search
            return self.traditional_search(query, top_k)
    
    def parse_date_query(self, query: str) -> Dict[str, Optional[str]]:
        """Extract date information from query."""
        if not self.llama_agent:
            return {"start_date": None, "end_date": None}
        
        try:
            response = self.llama_agent.client.inference.chat_completion(
                model=self.llama_agent.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts date information from queries."},
                    {"role": "user", "content": f"Extract the date information from this query: '{query}'. Return a JSON with start_date and end_date in YYYY-MM-DD format. If no specific end date is mentioned, use the end of the mentioned period."}
                ],
                stream=False
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                try:
                    date_info = json.loads(json_match.group(0))
                    return date_info
                except:
                    pass
        except Exception as e:
            print(f"Error parsing date query: {str(e)}")
        
        # Fallback to default date range
        return {"start_date": None, "end_date": None}
    
    def is_date_in_range(self, date_str: str, date_range: Dict[str, str]) -> bool:
        """Check if a date is within the specified range."""
        if not date_range["start_date"]:
            return False
        
        try:
            img_date = datetime.strptime(date_str, '%Y-%m-%d')
            start_date = datetime.strptime(date_range["start_date"], '%Y-%m-%d')
            
            if date_range["end_date"]:
                end_date = datetime.strptime(date_range["end_date"], '%Y-%m-%d')
                return start_date <= img_date <= end_date
            else:
                return img_date >= start_date
        except:
            return False
    
    def filter_by_date(self, images: List[Dict], date_range: Dict[str, str]) -> List[Dict]:
        """Filter images by date range."""
        if not date_range["start_date"]:
            return images
        
        return [img for img in images if self.is_date_in_range(img['creation_date'], date_range)]
    
    def confirm_deletion(self, images: List[Dict], query: str) -> str:
        """Generate confirmation message for deletion using Llama."""
        if not self.llama_agent:
            # Fallback confirmation message
            return f"Are you sure you want to delete these {len(images)} images matching '{query}'?"
        
        # Format image list
        image_list = "\n".join([
            f"- {img['path']} (Caption: {img['caption']}, Date: {img['creation_date']})"
            for img in images
        ])
        
        # Ask Llama to generate confirmation
        try:
            response = self.llama_agent.client.inference.chat_completion(
                model=self.llama_agent.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that helps users manage their image files."},
                    {"role": "user", "content": f"Based on the query '{query}', I found these images:\n{image_list}\n\nGenerate a confirmation message asking if the user wants to delete these {len(images)} images. Mention the date range if applicable."}
                ],
                stream=False
            )
            return response.text
        except Exception as e:
            print(f"Error generating confirmation: {str(e)}")
            # Fallback confirmation message
            return f"Are you sure you want to delete these {len(images)} images matching '{query}'?"
