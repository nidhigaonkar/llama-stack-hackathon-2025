
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict
import json


class VectorStore:
   def __init__(self, persist_directory: str = "chroma_db"):
       self.persist_directory = persist_directory
       self.chroma_client = chromadb.Client(Settings(
           persist_directory=persist_directory,
           anonymized_telemetry=False
       ))
      
       # Create or get the collection
       self.collection = self.chroma_client.get_or_create_collection("images")
      
       # Initialize the embedding model
       self.embed_model = HuggingFaceEmbedding(
           model_name="sentence-transformers/all-mpnet-base-v2"
       )
      
       # Initialize the vector store
       self.vector_store = ChromaVectorStore(
           chroma_collection=self.collection,
           embed_model=self.embed_model
       )
      
       # Create the index
       self.index = VectorStoreIndex.from_vector_store(
           self.vector_store,
           embed_model=self.embed_model
       )


   def add_images(self, processed_images: List[Dict[str, str]]):
       """Add processed images to the vector store."""
       documents = []
      
       for img_data in processed_images:
           # Create a rich text description combining caption and metadata
           content = f"""
           Image Caption: {img_data['caption']}
           Creation Date: {img_data['creation_date']}
           Creation Time: {img_data['creation_time']}
           File Path: {img_data['path']}
           """
          
           # Create metadata
           metadata = {
               "path": img_data['path'],
               "creation_date": img_data['creation_date'],
               "creation_time": img_data['creation_time']
           }
          
           # Create document
           doc = Document(
               text=content,
               metadata=metadata
           )
           documents.append(doc)
      
       # Add documents to the index
       self.index.insert_nodes(documents)


   def search_images(self, query: str, top_k: int = 5) -> List[Dict]:
       """Search for images based on the query."""
       # Create a query engine
       query_engine = self.index.as_query_engine(
           similarity_top_k=top_k
       )
      
       # Get response
       response = query_engine.query(query)
      
       # Extract and format results
       results = []
       for node in response.source_nodes:
           results.append({
               'path': node.metadata['path'],
               'creation_date': node.metadata['creation_date'],
               'creation_time': node.metadata['creation_time'],
               'relevance_score': node.score if hasattr(node, 'score') else None
           })
      
       return results

