import pandas as pd
import numpy as np
import os
try:
    from pipelines.config import (
        TRAFFY_CLEANED_PATH, 
        PM25_DATA_PATH, 
        DEPARTMENT_DATA_PATH, 
        RAINFALL_DATA_PATH,
        TRAFFY_MERGED_PATH
    )
except ImportError:
    from config import (
        TRAFFY_CLEANED_PATH, 
        PM25_DATA_PATH, 
        DEPARTMENT_DATA_PATH, 
        RAINFALL_DATA_PATH,
        TRAFFY_MERGED_PATH
    )


class FeatureEngineer:
    def __init__(self):
        pass

    def load_datasets(self):
        """Loads cleaned traffy data and external datasets."""
        print("Loading datasets for feature engineering...")
        try:
            df_traffy = pd.read_csv(TRAFFY_CLEANED_PATH)
            df_pm = pd.read_csv(PM25_DATA_PATH)
            df_dept = pd.read_csv(DEPARTMENT_DATA_PATH)
            df_rain = pd.read_csv(RAINFALL_DATA_PATH)
            
            print(f"Traffy shape: {df_traffy.shape}")
            print(f"PM2.5 shape: {df_pm.shape}")
            print(f"Department shape: {df_dept.shape}")
            print(f"Rainfall shape: {df_rain.shape}")
            
            return df_traffy, df_pm, df_dept, df_rain
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")
            return None, None, None, None

    def preprocess_dates(self, df_traffy, df_pm, df_rain):
        """Preprocesses date columns for merging."""
        print("Preprocessing dates...")
    
        # Traffy timestamps (robust parse, force UTC)
        df_traffy["timestamp"] = pd.to_datetime(
            df_traffy["timestamp"],
            format="mixed",
            errors="coerce",
            utc=True,           # datetime64[ns, UTC]
        )

        invalid_before = df_traffy["timestamp"].isna().sum()
        if invalid_before > 0:
            print(f"Dropping {invalid_before} rows with invalid timestamps in Traffy...")

        # ทำ copy หลัง dropna เพื่อลด SettingWithCopyWarning
        df_traffy = df_traffy.dropna(subset=["timestamp"]).copy()

        # Create date column (floor to date, still in UTC)
        df_traffy["date"] = df_traffy["timestamp"].dt.floor("D")
    
        # PM2.5 dates → UTC
        df_pm["date"] = pd.to_datetime(df_pm["date"], errors="coerce", utc=True)
        df_pm = df_pm.dropna(subset=["date"]).copy()
    
        # Rainfall dates → UTC
        df_rain["date"] = pd.to_datetime(df_rain["date"], errors="coerce", utc=True)
        df_rain = df_rain.dropna(subset=["date"]).copy()
    
        return df_traffy, df_pm, df_rain


    def merge_data(self, df_traffy, df_pm, df_rain, df_dept):
        """Merges Traffy with PM, rainfall, and department info."""
        print("Merging datasets...")
        
        # Merge with PM2.5
        df_merged = pd.merge(df_traffy, df_pm, on="date", how="inner")
        
        # Merge with Rainfall
        df_merged = pd.merge(df_merged, df_rain, on="date", how="inner")
        
        # Process Department Data (District mapping)
        district_map = {
            'Bang Bon': 'บางบอน', 'Bang Kapi': 'บางกะปิ', 'Bang Khae': 'บางแค', 
            'Bang Khen': 'บางเขน', 'Bang Kho Laem': 'บางคอแหลม', 'Bang Khun Thian': 'บางขุนเทียน',
            'Bang Na': 'บางนา', 'Bang Phlat': 'บางพลัด', 'Bang Rak': 'บางรัก', 
            'Bang Sue': 'บางซื่อ', 'Bangkok Noi': 'บางกอกน้อย', 'Bangkok Yai': 'บางกอกใหญ่',
            'Bueng Kum': 'บึงกุ่ม', 'Chatuchak': 'จตุจักร', 'Chom Thong': 'จอมทอง',
            'Din Daeng': 'ดินแดง', 'Don Mueang': 'ดอนเมือง', 'Dusit': 'ดุสิต',
            'Huai Khwang': 'ห้วยขวาง', 'Khan Na Yao': 'คันนายาว', 'Khlong Sam Wa': 'คลองสามวา',
            'Khlong San': 'คลองสาน', 'Khlong Toei': 'คลองเตย', 'Lak Si': 'หลักสี่',
            'Lat Krabang': 'ลาดกระบัง', 'Lat Phrao': 'ลาดพร้าว', 'Min Buri': 'มีนบุรี',
            'Nong Chok': 'หนองจอก', 'Nong Khaem': 'หนองแขม', 'Pathum Wan': 'ปทุมวัน',
            'Phasi Charoen': 'ภาษีเจริญ', 'Phaya Thai': 'พญาไท', 'Phra Khanong': 'พระโขนง',
            'Phra Nakhon': 'พระนคร', 'Pom Prap Sattru Phai': 'ป้อมปราบศัตรูพ่าย',
            'Prawet': 'ประเวศ', 'Rat Burana': 'ราษฎร์บูรณะ', 'Ratchathewi': 'ราชเทวี',
            'Sai Mai': 'สายไหม', 'Samphanthawong': 'สัมพันธวงศ์', 'Saphan Sung': 'สะพานสูง',
            'Sathon': 'สาทร', 'Suan Luang': 'สวนหลวง', 'Taling Chan': 'ตลิ่งชัน',
            'Thawi Watthana': 'ทวีวัฒนา', 'Thon Buri': 'ธนบุรี', 'Thung Khru': 'ทุ่งครุ',
            'Wang Thonglang': 'วังทองหลาง', 'Watthana': 'วัฒนา', 'Yan Nawa': 'ยานนาวา'
        }
        
        df_dept["district_thai"] = df_dept["district"].map(district_map)
        
        # Aggregate department counts by district
        dept_counts = df_dept.groupby("district_thai").size().reset_index(name="dept_count")
        
        # Merge department counts into Traffy
        df_merged = pd.merge(
            df_merged,
            dept_counts,
            left_on="district",
            right_on="district_thai",
            how="left",
        )
        
        print(f"Merged shape: {df_merged.shape}")
        return df_merged

    def finalize_features(self, df):
            """
            Final cleanup and feature selection:
            - drop unnamed cols
            - keep only completed state
            - create resolution_time (hours)
            - remove invalid / extreme resolution_time (trim)
            - one-hot encode type_clean
            - encode district as integer codes (with mapping)
            - drop organization (not used)
            """
            print("Finalizing features...")
    
            # Drop Unnamed columns if any
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    
            # Filter for completed state
            if "state" in df.columns:
                before_state = df.shape[0]
                df = df[df["state"] == "เสร็จสิ้น"].copy()
                print(f"Filtered state == 'เสร็จสิ้น': {before_state} -> {df.shape[0]} rows")
            else:
                print("WARNING: 'state' column not found; skipping state filter.")
    
            # Fill missing dept_count with 0
            if "dept_count" in df.columns:
                df["dept_count"] = df["dept_count"].fillna(0)
            else:
                print("WARNING: 'dept_count' column not found; will be missing in model features.")
    
            # --- Create resolution_time (hours) ---
            print("Creating resolution_time feature...")
    
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["last_activity_dt"] = pd.to_datetime(df["last_activity"], errors="coerce")
    
            invalid_ts = df["timestamp_dt"].isna().sum() + df["last_activity_dt"].isna().sum()
            if invalid_ts > 0:
                print(f"Dropping {invalid_ts} rows with invalid timestamp/last_activity...")
            df = df.dropna(subset=["timestamp_dt", "last_activity_dt"])
    
            df["resolution_time"] = (
                (df["last_activity_dt"] - df["timestamp_dt"]).dt.total_seconds() / 3600.0
            )
    
            # Filter out negative or unreasonable values
            before_rt = df.shape[0]
            df = df[(df["resolution_time"] > 0) & (df["resolution_time"] < 8760)]
            print(f"Filtered invalid resolution_time: {before_rt} -> {df.shape[0]} rows")
    
            # Trim (10%–75%)
            q_low, q_high = df["resolution_time"].quantile([0.10, 0.75])
            before_trim = df.shape[0]
            df = df[(df["resolution_time"] >= q_low) & (df["resolution_time"] <= q_high)]
            print(
                f"Trimmed resolution_time 10%-75% range "
                f"[{q_low:.2f}, {q_high:.2f}]: {before_trim} -> {df.shape[0]} rows"
            )
    
            # ---- ❌ Drop organization (not used) ----
            if "organization" in df.columns:
                df = df.drop(columns=["organization"])
                print("Dropped 'organization' column.")
    
            # ---- 🔠 One-hot encode type_clean ----
            if "type_clean" in df.columns:
                print("One-hot encoding 'type_clean'...")
                df["type_clean"] = df["type_clean"].fillna("Unknown").astype(str)
    
                type_dummies = pd.get_dummies(
                    df["type_clean"],
                    prefix="type",
                    dtype="int8"
                )
                print(f"Created {type_dummies.shape[1]} one-hot columns for 'type_clean'.")
                df = pd.concat([df, type_dummies], axis=1)
            else:
                print("WARNING: 'type_clean' column not found; skipping one-hot encoding.")
    
            # ---- 🔢 Encode district as integer codes (keep behavior) ----
            mapping_dir = os.path.join(os.path.dirname(TRAFFY_MERGED_PATH), "mappings")
            os.makedirs(mapping_dir, exist_ok=True)
    
            col_name = "district"
            if col_name in df.columns:
                print(f"Encoding categorical column '{col_name}' as integer codes...")
                df[col_name] = df[col_name].fillna("Unknown").astype(str)
                codes, uniques = pd.factorize(df[col_name])
                df[f"{col_name}_code"] = codes.astype("int32")
    
                # Build mapping: value -> code
                mapping = dict(zip(uniques, range(len(uniques))))
    
                print(f"\n=== Mapping for '{col_name}' (value -> {col_name}_code) ===")
                for i, (val, code) in enumerate(mapping.items()):
                    if i >= 30:
                        print("... (truncated)")
                        break
                    print(f"{repr(val)} -> {code}")
                print("=============================================\n")
    
                # Save mapping to CSV
                mapping_df = pd.DataFrame(
                    {
                        col_name: list(mapping.keys()),
                        f"{col_name}_code": list(mapping.values()),
                    }
                )
                mapping_path = os.path.join(mapping_dir, f"{col_name}_mapping.csv")
                mapping_df.to_csv(mapping_path, index=False)
                print(f"Saved {col_name} mapping to: {mapping_path}")
            else:
                print(f"WARNING: categorical column '{col_name}' not found; skipping encoding.")
    
            # Drop helper datetime columns
            df = df.drop(columns=["timestamp_dt", "last_activity_dt"], errors="ignore")
    
            print(f"Final shape: {df.shape}")
            return df



    def run(self):
        df_traffy, df_pm, df_dept, df_rain = self.load_datasets()
        
        if df_traffy is not None:
            df_traffy, df_pm, df_rain = self.preprocess_dates(df_traffy, df_pm, df_rain)
            df_merged = self.merge_data(df_traffy, df_pm, df_rain, df_dept)
            df_final = self.finalize_features(df_merged)
            
            # Save model-ready dataset
            df_final.to_csv(TRAFFY_MERGED_PATH, index=False)
            print(f"Merged data saved to {TRAFFY_MERGED_PATH}")
            return df_final
        return None


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run()
