import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
from collections import Counter

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load product data
df1 = pd.read_csv("data/dresses_bd_processed_data.csv")
df2 = pd.read_csv("data/jeans_bd_processed_data.csv")
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=['product_id', 'product_name', 'feature_image_s3'])
df['product_id'] = df['product_id'].astype(str)
df = df.drop_duplicates(subset='product_id', keep='first')
product_info = df.set_index('product_id').to_dict(orient='index')

# Load embeddings
embeddings = np.load("embeddings/image_features.npy")
image_paths = np.load("embeddings/image_ids.npy")
image_paths = [x.decode("utf-8") if isinstance(x, bytes) else x for x in image_paths]

# Remove duplicates
seen = set()
unique_indices = [i for i, path in enumerate(image_paths) if path not in seen and not seen.add(path)]
embeddings = embeddings[unique_indices]
image_paths = [image_paths[i] for i in unique_indices]

# Session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# UI setup
st.set_page_config(page_title="StyleSearch - Your AI Fashion Assistant", layout="wide")
st.markdown("""
    <style>
    .hero {background: linear-gradient(to right, #fff, #f3f3f3); padding: 40px 20px; text-align: center;
    border-radius: 12px; margin-bottom: 40px; box-shadow: 0 6px 24px rgba(0,0,0,0.05);}
    .hero h1 {font-size: 3.5rem; font-weight: 700; color: #111;}
    .hero p {font-size: 1.2rem; color: #555; margin-top: 10px;}
    .product-card {background-color: #fff; padding: 20px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); 
    text-align: center; transition: transform 0.3s ease;}
    .product-card:hover {transform: translateY(-5px); box-shadow: 0 12px 32px rgba(0,0,0,0.15);}
    .product-name {font-size: 1.1rem; font-weight: 600; color: #222; margin-top: 10px;}
    .brand-name {font-size: 0.95rem; color: #999;}
    .price-tag {font-size: 1.5rem; font-weight: 700; color: #D6336C; background-color: #FFF0F5; 
    padding: 8px 16px; display: inline-block; border-radius: 12px; margin-top: 10px;}
    .style-summary {font-size: 0.85rem; color: #666; margin-top: 6px; font-style: italic; text-align: left;}
    hr {border: none; border-top: 1px solid #ddd; margin: 40px 0;}
    </style>
    <div class='hero'>
        <h1>StyleSearch</h1>
        <p>Discover Your Look. Powered by AI. Styled by You.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Upload a fashion image", type=["jpg", "jpeg", "png"])

# Helper functions
def extract_style_tags(entry):
    tags = []
    for key in ['meta_info', 'style_attributes', 'feature_list']:
        raw = entry.get(key)
        if not raw: continue
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except: parsed = raw
        if isinstance(parsed, dict):
            tags.extend([f"{k} {v}" for k, v in parsed.items()])
        elif isinstance(parsed, list):
            tags.extend([str(t).strip() for t in parsed])
        elif isinstance(parsed, str):
            tags.extend(re.split(r"[,\[\]\n]", raw))
    return [re.sub(r"[\"'\[\]{}]", "", tag.strip()) for tag in tags if tag and len(tag.strip()) > 2]

def generate_heuristic_summary(tags):
    if not tags:
        return "<ul><li>Style details not available.</li></ul>"
    seen = set()
    filtered = []
    for tag in tags:
        if tag in seen: continue
        seen.add(tag)
        filtered.append(tag)
    if not filtered:
        return "<ul><li>Gentle care instructions. Designed for comfort and style.</li></ul>"
    bullet_items = "".join(f"<li>{t.capitalize()}</li>" for t in filtered[:6])
    return f"<ul style='padding-left: 20px; margin: 0;'>{bullet_items}</ul>"

def detect_category(text):
    if not isinstance(text, str): return "misc"
    t = text.lower()
    if any(k in t for k in ["top", "shirt", "blouse"]): return "top"
    if any(k in t for k in ["jean", "pant", "bottom"]): return "bottom"
    if "dress" in t: return "dress"
    if any(k in t for k in ["shoe", "heel"]): return "shoes"
    if "bag" in t: return "bag"
    return "misc"

def calculate_tag_overlap(tags1, tags2):
    return len(set(tags1) & set(tags2))

def format_price(price):
    try:
        if isinstance(price, str) and ("INR" in price or "USD" in price):
            price = ast.literal_eval(price)
        if isinstance(price, dict):
            if 'INR' in price:
                return f"₹{price['INR']:,.0f}"
            elif 'USD' in price:
                return f"${price['USD']:,.2f}"
        elif isinstance(price, (int, float)):
            return f"₹{price:,.0f}"
    except: pass
    return "N/A"

# Main logic
if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Your Uploaded Item", width=320)
    query_tensor = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(query_tensor).cpu().numpy()
    st.session_state.search_history.append(query_embedding)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1]
    shown = set()
    st.markdown("## 🎯 Top Visual Matches")
    cols = st.columns(4)
    count = 0

    for idx in top_indices:
        img_path = image_paths[idx]
        img_id = img_path.replace(".jpg", "")
        full_path = f"images/{img_path}"
        if img_path in shown or not os.path.exists(full_path): continue
        shown.add(img_path)
        data = product_info.get(img_id, {})
        with cols[count % 4]:
            st.markdown("<div class='product-card'>", unsafe_allow_html=True)
            st.image(full_path, use_column_width=True)
            st.markdown(f"<div class='product-name'>{data.get('product_name', 'Unknown')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='brand-name'>{data.get('brand', 'Brand')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='price-tag'>{format_price(data.get('selling_price'))}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='style-summary'>{generate_heuristic_summary(extract_style_tags(data))}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        count += 1
        if count >= 8: break

    st.markdown("<hr><h3 style='text-align:center'>🧥 Recommended Outfit</h3>", unsafe_allow_html=True)
    best_id = image_paths[top_indices[0]].replace(".jpg", "")
    best_data = product_info.get(best_id, {})
    if not best_data:
        st.warning("Could not find details for the best match. Try a different image.")
    else:
        query_cat = detect_category(best_data.get("description") or best_data.get("product_name") or "")
        best_tags = extract_style_tags(best_data)
        comp_map = {
            "top": ["bottom", "bag"],
            "bottom": ["top", "bag"],
            "dress": ["shoes", "bag"],
            "shoes": ["dress"],
            "bag": ["dress", "top"]
        }
        comps = comp_map.get(query_cat, ['dress', 'top', 'bottom'])
        ranked = []
        for pid, info in product_info.items():
            img_file = f"{pid}.jpg"
            if img_file in shown or not os.path.exists(f"images/{img_file}"): continue
            if detect_category(info.get("product_name", "")) not in comps: continue
            overlap = calculate_tag_overlap(best_tags, extract_style_tags(info))
            try:
                sim = cosine_similarity(query_embedding, embeddings[image_paths.index(img_file)].reshape(1, -1))[0][0]
            except:
                sim = 0
            score = 0.7 * overlap + 0.3 * sim
            ranked.append((score, pid))
            shown.add(img_file)

        ranked.sort(reverse=True)
        cols = st.columns(4)
        for i, (_, pid) in enumerate(ranked[:4]):
            full_path = f"images/{pid}.jpg"
            data = product_info[pid]
            with cols[i % 4]:
                st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                st.image(full_path, use_column_width=True)
                st.markdown(f"<div class='product-name'>{data.get('product_name', 'Unknown')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='brand-name'>{data.get('brand', 'Brand')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='price-tag'>{format_price(data.get('selling_price'))}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='style-summary'>{generate_heuristic_summary(extract_style_tags(data))}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Trending Now section
    st.markdown("## 🔥 Trending Now")
    st.caption("📈 Trends are inferred from tag frequency across the catalog.")
    theme_keywords = [
        ("Soft Neutrals", ["beige", "nude", "taupe", "ivory"]),
        ("Bold Florals", ["floral", "petal", "bloom", "rose"]),
        ("Monochrome Magic", ["black", "white", "gray", "monochrome"]),
        ("Denim Revival", ["denim", "jeans", "washed", "blue"]),
        ("Evening Elegance", ["satin", "gown", "evening", "lace"]),
        ("Everyday Comfort", ["cotton", "casual", "relaxed", "stretch"])
    ]
    theme_descriptions = {
        "Soft Neutrals": "Subtle tones like beige and ivory for minimalist chic.",
        "Bold Florals": "Lively blooms to brighten up spring and summer.",
        "Monochrome Magic": "Black, white and grey never go out of style.",
        "Denim Revival": "The return of timeless blue denim in all forms.",
        "Evening Elegance": "Elegant picks for special evenings.",
        "Everyday Comfort": "Relaxed, breathable everyday wear."
    }
    all_tags = []
    for entry in product_info.values():
        all_tags += extract_style_tags(entry)
    tag_counter = Counter([t.lower().strip() for t in all_tags if len(t) > 2])
    matched_themes = []
    for name, keywords in theme_keywords:
        count = sum(tag_counter[k] for k in keywords if k in tag_counter)
        if count > 0:
            matched_themes.append((name, count))
    matched_themes.sort(key=lambda x: -x[1])
    for name, count in matched_themes[:5]:
        st.markdown(f"- 🌟 **{name}** — Loved by {count} fashionistas")
        st.caption(f"  {theme_descriptions.get(name, '')}")

    with st.expander("⚙️ Architecture Plan for Scaling"):
        st.markdown("""
        - **Frontend**: Streamlit / React + FastAPI  
        - **Backend**: TorchServe, FAISS, Redis  
        - **Vector DB**: Milvus for ANN search  
        - **Infra**: Dockerized on AWS ECS / Kubernetes  
        - **Caching**: Redis for recent results  
        - **Style Trends**: Daily scraping & clustering from fashion sources
        """)
else:
    st.info("📸 Upload a fashion image to explore similar styles and outfit matches.")
