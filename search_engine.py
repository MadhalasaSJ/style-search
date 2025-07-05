import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load stored embeddings
embeddings = np.load("embeddings/image_features.npy")
image_paths = np.load("embeddings/image_ids.npy")

# Converting bytes to string
image_paths = [x.decode("utf-8") if isinstance(x, bytes) else x for x in image_paths]

# Query image path
query_image_path = r"C:\Users\madha\VisualStudio\fashion-visual-search\images\0a0e1710dcdddf87624fc1e55a9d58385342f388c0692ea3ab9abb9e4af203d7.jpg"
query_image = preprocess(Image.open(query_image_path)).unsqueeze(0).to(device)

# Generate embedding for query
with torch.no_grad():
    query_embedding = model.encode_image(query_image).cpu().numpy()

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]
top_k = 5
top_indices = similarities.argsort()[-top_k:][::-1]

# Show results
print("Top matches:")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {image_paths[idx]} (Score: {similarities[idx]:.4f})")
