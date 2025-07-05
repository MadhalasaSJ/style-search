from utils import load_dataset, download_images
import pandas as pd

# Load both datasets
df1 = load_dataset(r"C:\Users\madha\VisualStudio\fashion-visual-search\data\dresses_bd_processed_data.csv")
df2 = load_dataset(r"C:\Users\madha\VisualStudio\fashion-visual-search\data\jeans_bd_processed_data.csv")

# Combine them
combined_df = pd.concat([df1, df2], ignore_index=True)

# Download images 
download_images(combined_df, folder='images')