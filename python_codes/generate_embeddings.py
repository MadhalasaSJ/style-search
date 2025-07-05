import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder with all images
image_folder = 'images'
embeddings = []
image_ids = []

for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            embedding = embedding.cpu().numpy().flatten()
            embeddings.append(embedding)
            image_ids.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save embeddings and IDs
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/image_features.npy", np.array(embeddings))
np.save("embeddings/image_ids.npy", np.array(image_ids))

print("✅ Image embeddings saved.")
