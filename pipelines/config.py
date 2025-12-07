
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File Paths - Raw
TRAFFY_RAW_PATH = os.path.join(RAW_DATA_DIR, "bangkok_traffy.csv")
DEPARTMENT_DATA_PATH = os.path.join(RAW_DATA_DIR, "department_data.csv") # Assuming scraped data goes here
PM25_DATA_PATH = os.path.join(RAW_DATA_DIR, "pm25_data.csv")
RAINFALL_DATA_PATH = os.path.join(RAW_DATA_DIR, "rainfall_data.csv")

# File Paths - Processed
TRAFFY_CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, "bangkok_traffy_cleaned.csv") # Intermediate
TRAFFY_MERGED_PATH = os.path.join(PROCESSED_DATA_DIR, "bangkok_traffy_merged.csv")
MODEL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "eiei.csv") # From modeling.ipynb

# Scraping Config
SOIDB_URLS = [
    "https://www.soidb.com/bangkok/fire/list.html",
    "https://www.soidb.com/bangkok/police/list.html",
    "https://www.soidb.com/bangkok/government/list.html"
]
PM25_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
RAINFALL_API_URL = "https://archive-api.open-meteo.com/v1/archive"
METEOSTAT_BASE_URL = "https://meteostat.p.rapidapi.com/stations/daily" # From notebook, though logic used Open-Meteo

# Coordinates for Bangkok
LATITUDE = 13.7563
LONGITUDE = 100.5018
TIMEZONE = "Asia/Bangkok"
