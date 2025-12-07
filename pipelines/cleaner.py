
import pandas as pd
import numpy as np
import os
try:
    from pipelines.config import TRAFFY_RAW_PATH, TRAFFY_CLEANED_PATH
except ImportError:
    from config import TRAFFY_RAW_PATH, TRAFFY_CLEANED_PATH

class Cleaner:
    def __init__(self):
        pass

    def load_data(self):
        """Loads raw Traffy Fondue data."""
        print(f"Loading raw data from {TRAFFY_RAW_PATH}...")
        try:
            df = pd.read_csv(TRAFFY_RAW_PATH, parse_dates=["timestamp", "last_activity"], low_memory=False)
            print(f"Loaded df shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {TRAFFY_RAW_PATH}")
            return None

    def clean_traffy_data(self, df):
        """Performs initial data cleaning and feature extraction."""
        if df is None:
            return None
    
        print("Cleaning Traffy data...")
    
        # Clean 'type' column → keep only the first type
        df["type_clean"] = (
            df["type"]
            .astype(str)
            .str.strip()
            .str.strip("{}[]")                 # ตัด { } [ ] รอบนอก
            .str.replace("'", "", regex=False) # ตัด '
            .str.replace('"', "", regex=False) # ตัด "
            .str.split(",")
            .str[0]                            # เอาอันแรกอย่างเดียว
            .str.strip()
        )
    
        # Create 'has_photo' feature
        df["has_photo"] = df["photo"].notnull() | df["photo_after"].notnull()
    
        # Calculate 'comment_len'
        df["comment_len"] = df["comment"].fillna("").str.len()
    
        # Extract coordinates
        coords = df["coords"].astype(str).str.split(",", expand=True)
        if coords.shape[1] >= 2:
            df["lon"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
            df["lat"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")
    
        print(f"Cleaned df shape: {df.shape}")
        return df


    def save_clean_data(self, df):
        """Saves cleaned data to CSV."""
        if df is not None:
            df.to_csv(TRAFFY_CLEANED_PATH, index=False)
            print(f"Cleaned data saved to {TRAFFY_CLEANED_PATH}")

    def run(self):
        df = self.load_data()
        df_clean = self.clean_traffy_data(df)
        self.save_clean_data(df_clean)
        return df_clean

if __name__ == "__main__":
    cleaner = Cleaner()
    cleaner.run()
