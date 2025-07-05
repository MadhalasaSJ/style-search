# 👗 StyleSearch – Visual Fashion Search and Outfit Recommender

StyleSearch is a machine learning-powered visual search tool that helps users find fashion items similar to an uploaded image. It also suggests matching outfits using a blend of computer vision and metadata understanding.

---

## 🚀 How to Run

1. Make sure Python **3.8+** is installed.

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Launch the Streamlit app:

   * streamlit run app.py

## ✅ Ensure the following folders are present alongside the code:

   - data/ (e.g., metadata CSVs)

   - embeddings/ (precomputed image embeddings)

   - images/ (downloaded fashion images)


## 💡 About the Project

- Upload any fashion image.

- The system finds visually similar products using MobileNet (or CLIP) and cosine similarity.

- It recommends matching outfits using multi-modal similarity (visual + metadata).

- Personalized suggestions improve as the user interacts.

- A trend-aware module highlights top fashion themes based on search patterns.

## 🗂️ Project Structure

.
├── app.py                  # Main Streamlit app

├── generate_embeddings.py  # Embedding generator using MobileNet

├── search_engine.py        # Cosine similarity-based search logic

├── download_images.py      # Utility to download product images

├── utils.py                # Helper functions

├── data/                   # Metadata files

├── embeddings/             # Image embeddings

├── images/                 # Downloaded product images

├── requirements.txt

└── README.md




## ✨ Highlights
- 📸 Visual search with real-time feedback

- 🤖 Lightweight CNN-based embedding for efficient similarity

- 🎯 Trend awareness and personalization (optional modules)

- ⚡ Fast, simple UI built using Streamlit
