import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import numpy as np
import os
import sys

BANGKOK_DISTRICTS = {
    "บางรัก": (13.7248, 100.5265),
    "ปทุมวัน": (13.7469, 100.5362),
    "ดุสิต": (13.7777, 100.5155),
    "บางกอกน้อย": (13.7681, 100.4844),
    "บางกอกใหญ่": (13.7294, 100.4989),
    "บางขุนเทียน": (13.6469, 100.4230),
    "บางเขน": (13.8868, 100.6072),
    "บางแค": (13.7053, 100.3960),
    "บางคอแหลม": (13.7066, 100.5155),
    "บางซื่อ": (13.8032, 100.5345),
    "บางนา": (13.6670, 100.6099),
    "บางบอน": (13.6640, 100.3952),
    "บางพลัด": (13.7859, 100.4980),
    "บางระหว่าง": (13.6775, 100.5120),
    "บางกะปิ": (13.7622, 100.6424),
    "ภาษีเจริญ": (13.7310, 100.4413),
    "พญาไท": (13.7697, 100.5446),
    "ป้อมปราบศัตรูพ่าย": (13.7538, 100.5131),
    "ประเวศ": (13.6924, 100.6726),
    "ราชเทวี": (13.7507, 100.5349),
    "ราษฎร์บูรณะ": (13.6811, 100.5103),
    "ลาดกระบัง": (13.7285, 100.7489),
    "ลาดพร้าว": (13.8165, 100.6051),
    "วังทองหลาง": (13.7759, 100.5967),
    "วัฒนา": (13.7236, 100.5842),
    "สะพานสูง": (13.8222, 100.6657),
    "สวนหลวง": (13.7367, 100.6473),
    "สาทร": (13.7192, 100.5319),
    "สายไหม": (13.9015, 100.6542),
    "สัมพันธวงศ์": (13.7400, 100.5123),
    "คลองเตย": (13.7221, 100.5686),
    "คลองสาน": (13.7246, 100.5067),
    "คลองสามวา": (13.8448, 100.7204),
    "คันนายาว": (13.8298, 100.6976),
    "จตุจักร": (13.8155, 100.5542),
    "จอมทอง": (13.6663, 100.4538),
    "ดอนเมือง": (13.9174, 100.5976),
    "ดินแดง": (13.7664, 100.5586),
    "ทวีวัฒนา": (13.7731, 100.3655),
    "ทุ่งครุ": (13.6285, 100.5040),
    "ธนบุรี": (13.7300, 100.4893),
    "บึงกุ่ม": (13.8086, 100.6438),
    "มีนบุรี": (13.8120, 100.7424),
    "ยานนาวา": (13.6965, 100.5385),
    "หนองแขม": (13.6928, 100.3444),
    "หนองจอก": (13.8567, 100.8384),
    "หลักสี่": (13.8722, 100.5736),
    "ห้วยขวาง": (13.7747, 100.5819),
}

# Initialize Spark Session
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("TraffyFonduePredictionApp") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

# Load Model
@st.cache_resource
def load_model(_spark):
    model_path = "models/xgboost_spark_model"
    if os.path.exists(model_path):
        return PipelineModel.load(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please train and save the model first.")
        return None

st.title("Traffy Fondue Resolution Time Prediction")
st.write("Predict the time required to resolve a complaint based on its details.")

spark = get_spark_session()
model = load_model(spark)

if model:
    # Select district outside the form to trigger updates
    district = st.selectbox("District", list(BANGKOK_DISTRICTS.keys()))
    default_lat, default_lon = BANGKOK_DISTRICTS[district]

    with st.form("prediction_form"):
        st.subheader("Complaint Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            organization = st.text_input("Organization", value=f"เขต{district}")
            type_clean = st.selectbox("Type", ["ความสะอาด", "ทางเท้า", "แสงสว่าง", "ถนน", "น้ำท่วม", "อื่นๆ"])
            # district is selected outside
            timestamp = st.date_input("Date", value=pd.to_datetime("today"))
            hour = st.slider("Hour", 0, 23, 12)
            
        with col2:
            lat = st.number_input("Latitude", value=default_lat)
            lon = st.number_input("Longitude", value=default_lon)
            comment_len = st.number_input("Comment Length", value=50)
            count_reopen = st.number_input("Reopen Count", value=0)
            
        st.subheader("Environmental Factors")
        col3, col4 = st.columns(2)
        
        with col3:
            rainfall_mm = st.number_input("Rainfall (mm)", value=0.0)
            has_rain = st.checkbox("Is Raining?", value=False)
            
        with col4:
            pm25_avg = st.number_input("PM2.5", value=20.0)
            pm10_avg = st.number_input("PM10", value=40.0)
            dust_avg = st.number_input("Dust", value=30.0)

        # Hidden/Default fields required by the model
        # Note: The model was trained with 'last_activity' and 'timestamp'. 
        # In a real prediction scenario, we don't know 'last_activity' (closing time).
        # We will set 'last_activity' to 'timestamp' as a placeholder, 
        # but be aware this might affect prediction accuracy if the model relies heavily on it.
        
        submitted = st.form_submit_button("Predict Resolution Time")

    if submitted:
        # Prepare input data
        # Construct timestamp string
        timestamp_str = f"{timestamp} {hour}:00:00"
        
        data = {
            'organization': [organization],
            'type_clean': [type_clean],
            'district': [district],
            'coords': [f"{lon},{lat}"],
            'timestamp': [timestamp_str],
            'state': ['เสร็จสิ้น'], # Default state
            'count_reopen': [count_reopen],
            'last_activity': [timestamp_str], # Placeholder
            'comment_len': [comment_len],
            'lon': [lon],
            'lat': [lat],
            'pm25_avg': [pm25_avg],
            'pm10_avg': [pm10_avg],
            'dust_avg': [dust_avg],
            'rainfall_mm': [rainfall_mm],
            'has_rain': [str(has_rain).lower()], # Model might expect string or boolean depending on CSV reading
            'dept_count': [1] # Default
        }
        
        # Create Spark DataFrame
        input_df = spark.createDataFrame(pd.DataFrame(data))
        
        # Force types to match modeling.ipynb exactly
        # In modeling.ipynb, all feature_cols were explicitly cast to double.
        from pyspark.sql.functions import col
        
        feature_cols = [
           'coords','timestamp',
           'state', 'count_reopen', 'last_activity', 
           'comment_len', 'lon', 'lat','pm25_avg',
           'pm10_avg', 'dust_avg', 'rainfall_mm', 'has_rain', 'dept_count']
           
        categorical_cols = ['organization',"type_clean", "district"]
        
        select_exprs = []
        
        # Cast feature columns to Double
        for c in feature_cols:
            if c in ['timestamp', 'last_activity']:
                # Convert String -> Timestamp -> Double (Epoch)
                select_exprs.append(col(c).cast("timestamp").cast("double").alias(c))
            else:
                # Convert String/Int/Float -> Double
                select_exprs.append(col(c).cast("double").alias(c))
                
        # Keep categorical columns as is (String)
        for c in categorical_cols:
            select_exprs.append(col(c))
            
        # Apply selection and casting
        input_df = input_df.select(select_exprs)
        
        # Fill NA with 0 (matches training Imputer/casting behavior)
        input_df = input_df.fillna(0)

        # Predict
        try:
            prediction = model.transform(input_df)

            # Get result (log transformed)
            log_pred = prediction.select("prediction").collect()[0][0]
            
            # Inverse Log Transform (expm1)
            hours_pred = np.expm1(log_pred)
            
            st.success(f"Predicted Resolution Time: {hours_pred:.2f} hours")
            
            if hours_pred > 24:
                st.info(f"({hours_pred/24:.1f} days)")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Debug Info:")
            st.write(data)
