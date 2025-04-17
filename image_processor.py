import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import magic
from typing import List, Dict, Tuple

class ImageProcessor:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def is_png(self, file_path: str) -> bool:
        """Check if file is a PNG using python-magic."""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type == 'image/png'

    def scan_desktop(self) -> List[str]:
        """Scan desktop for PNG files."""
        desktop_path = str(Path.home() / "Desktop")
        png_files = []
        for root, _, files in os.walk(desktop_path):
            for file in files:
                if file.lower().endswith('.png'):
                    full_path = os.path.join(root, file)
                    if self.is_png(full_path):
                        png_files.append(full_path)
        return png_files

    def generate_caption(self, image_path: str) -> str:
        """Generate caption for an image using BLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return ""


    def process_single_image(self, image_path: str) -> Dict[str, str]:
        """Process a single image and return its metadata."""
        try:
            caption = self.generate_caption(image_path)
            creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
            return {
                'path': image_path,
                'caption': caption,
                'creation_date': creation_time.strftime('%Y-%m-%d'),
                'creation_time': creation_time.strftime('%H:%M:%S')
            }
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def scan_desktop(self) -> List[str]:
       """Scan desktop for PNG files."""
       desktop_path = str(Path.home() / "Desktop")
       png_files = []
      
       for root, _, files in os.walk(desktop_path):
           for file in files:
               if file.lower().endswith('.png'):
                   full_path = os.path.join(root, file)
                   if self.is_png(full_path):
                       png_files.append(full_path)
      
       return png_files

    def generate_caption(self, image_path: str) -> str:
       """Generate caption for an image using BLIP."""
       try:
           image = Image.open(image_path).convert('RGB')
           inputs = self.processor(image, return_tensors="pt").to(self.device)
          
           out = self.model.generate(**inputs, max_length=50)
           caption = self.processor.decode(out[0], skip_special_tokens=True)
          
           return caption
       except Exception as e:
           print(f"Error processing image {image_path}: {str(e)}")
           return ""


    def process_images(self) -> List[Dict[str, str]]:
       """Process all PNG images on desktop and return their metadata."""
       png_files = self.scan_desktop()
       processed_images = []
      
       for image_path in png_files:
           try:
               caption = self.generate_caption(image_path)
               creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
              
               processed_images.append({
                   'path': image_path,
                   'caption': caption,
                   'creation_date': creation_time.strftime('%Y-%m-%d'),
                   'creation_time': creation_time.strftime('%H:%M:%S')
               })
              
           except Exception as e:
               print(f"Error processing {image_path}: {str(e)}")
               continue
      
       return processed_images

    def build_caption_index(image_dir: str):
        documents = []
        for path in Path(image_dir).glob("*.png"):
            caption = generate_caption(str(path))
            text = f"{caption} ||| {path.name}"  # Append filename
            documents.append(Document(text=text))
        
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        return index
