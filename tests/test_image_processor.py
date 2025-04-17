from image_processor import ImageProcessor
import os

# Initialize the image processor
processor = ImageProcessor()

# Test with a sample PNG image
test_image_path = "/path/to/test/image.png"  # Replace with an actual PNG path
if os.path.exists(test_image_path):
    caption = processor.generate_caption(test_image_path)
    print(f"Image: {test_image_path}")
    print(f"Caption: {caption}")
    print(f"Is PNG: {processor.is_png(test_image_path)}")
else:
    print(f"Test image not found at {test_image_path}")
