import os
import requests
from tqdm import tqdm
import pandas as pd

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")

def download_images(df, folder='images'):
    os.makedirs(folder, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row['feature_image_s3']
        product_id = row['product_id']
        try:
            # Save as product_id.jpg
            image_path = os.path.join(folder, f"{product_id}.jpg")
            if not os.path.exists(image_path):
                response = requests.get(img_url, timeout=10)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
