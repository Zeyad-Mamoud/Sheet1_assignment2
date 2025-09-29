from transformers import ViTImageProcessor, ViTModel
from PIL import Image
#import torch
image = Image.open("1.jpg").convert("RGB")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# 3) Preprocess → Tensor
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

