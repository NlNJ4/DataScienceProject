
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import time
from urllib.parse import urljoin
import re
from datetime import datetime
import os
try:
    from pipelines.config import DEPARTMENT_DATA_PATH, PM25_DATA_PATH, RAINFALL_DATA_PATH, SOIDB_URLS, PM25_API_URL, RAINFALL_API_URL, LATITUDE, LONGITUDE, TIMEZONE
except ImportError:
    from config import DEPARTMENT_DATA_PATH, PM25_DATA_PATH, RAINFALL_DATA_PATH, SOIDB_URLS, PM25_API_URL, RAINFALL_API_URL, LATITUDE, LONGITUDE, TIMEZONE

class Scraper:
    def __init__(self):
        self.request_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape_soidb(self):
        """Scrapes department data from soidb.com."""
        print("Starting scraping from soidb.com...")
        headers = ["name", "road", "district", "province"]
        
        with open(DEPARTMENT_DATA_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for start_url in SOIDB_URLS:
                print(f"--- Starting Category: {start_url} ---")
                current_url = start_url
                
                while True:
                    print(f"Scraping: {current_url}")
                    try:
                        response = requests.get(current_url, headers=self.request_headers, timeout=10)
                        if response.status_code != 200:
                            print(f"Failed to load page: {response.status_code}")
                            break

                        soup = BeautifulSoup(response.content, "html.parser")
                        items = soup.find_all("div", class_="list_main")

                        if not items:
                            print("No items found on this page.")

                        for item in items:
                            name_tag = item.find("div", class_="sd_link", itemprop="name")
                            name = name_tag.get_text(strip=True) if name_tag else ""

                            road = ""
                            district = ""
                            province = ""

                            address_span = item.find("span", itemprop="address")
                            if address_span:
                                road_tag = address_span.find("span", itemprop="streetAddress")
                                if road_tag:
                                    road = road_tag.get_text(strip=True)
                                
                                district_tag = address_span.find("span", itemprop="addressLocality")
                                if district_tag:
                                    district = district_tag.get_text(strip=True)
                                
                                province_tag = address_span.find("span", itemprop="addressRegion")
                                if province_tag:
                                    province = province_tag.get_text(strip=True)

                            writer.writerow([name, road, district, province])

                        next_link = soup.find("a", string=re.compile(r"Next", re.IGNORECASE))
                        
                        if next_link and 'href' in next_link.attrs:
                            next_url = next_link['href']
                            current_url = urljoin(start_url, next_url)
                            time.sleep(1)
                        else:
                            print("No 'Next' page found. Moving to next category.")
                            break

                    except Exception as e:
                        print(f"Error occurred: {e}")
                        break
        
        print(f"\nDone! Data saved to {DEPARTMENT_DATA_PATH}")

    def fetch_pm25_data(self, start_date='2023-01-01', end_date='2025-01-31'):
        """Fetches PM2.5 data from Open-Meteo Air Quality API."""
        print("Fetching PM2.5 data from Open-Meteo Air Quality API...")
        
        params = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'pm2_5,pm10,dust',
            'timezone': TIMEZONE
        }
        
        try:
            print(f"  Requesting data from {start_date} to {end_date}...")
            response = requests.get(PM25_API_URL, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'hourly' in data:
                    df = pd.DataFrame({
                        'datetime': pd.to_datetime(data['hourly']['time']),
                        'pm25': data['hourly']['pm2_5'],
                        'pm10': data['hourly']['pm10'],
                        'dust': data['hourly']['dust']
                    })
                    
                    df['date'] = df['datetime'].dt.date
                    df_daily = df.groupby('date').agg({
                        'pm25': ['mean', 'max', 'min'],
                        'pm10': 'mean',
                        'dust': 'mean'
                    }).reset_index()
                    
                    df_daily.columns = ['date', 'pm25_avg', 'pm25_max', 'pm25_min', 'pm10_avg', 'dust_avg']
                    df_daily['date'] = pd.to_datetime(df_daily['date'])
                    
                    df_daily = df_daily.dropna(subset=['pm25_avg'])
                    
                    print(f"\n✓ PM2.5 data collected: {len(df_daily)} days")
                    print(f"  Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
                    print(f"  Avg PM2.5: {df_daily['pm25_avg'].mean():.2f} µg/m³")
                    
                    df_daily.to_csv(PM25_DATA_PATH, index=False)
                    print(f"PM2.5 data saved to {PM25_DATA_PATH}")
                    return df_daily
                else:
                    print("No air quality data in response")
                    return None
            else:
                print(f"  Error: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  Error fetching PM2.5 data: {e}")
            return None

    def fetch_rainfall_data(self, start_date='2021-01-01', end_date='2025-01-31'):
        """Fetches rainfall data from Open-Meteo Archive API."""
        print("Fetching rainfall data...")
        
        params = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'precipitation_sum,precipitation_hours',
            'timezone': TIMEZONE
        }
        
        try:
            response = requests.get(RAINFALL_API_URL, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                
                if 'daily' in data:
                    df = pd.DataFrame({
                        'date': pd.to_datetime(data['daily']['time']),
                        'rainfall_mm': data['daily']['precipitation_sum'],
                        'rainfall_hours': data['daily']['precipitation_hours']
                    })
                    
                    df['has_rain'] = (df['rainfall_mm'] > 0).astype(int)
                    df['heavy_rain'] = (df['rainfall_mm'] > 35).astype(int)
                    
                    print(f"\n✓ Rainfall data collected: {len(df)} days")
                    
                    df.to_csv(RAINFALL_DATA_PATH, index=False)
                    print(f"Rainfall data saved to {RAINFALL_DATA_PATH}")
                    return df
            else:
                print(f"Error: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching rainfall data: {e}")
            return None

    def run_all(self):
        self.scrape_soidb()
        self.fetch_pm25_data()
        self.fetch_rainfall_data()

if __name__ == "__main__":
    scraper = Scraper()
    scraper.run_all()
