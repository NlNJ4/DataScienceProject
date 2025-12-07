import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pydeck as pdk

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Traffy Fondue Analytics",
    page_icon="🏙️",
    layout="wide"
)
dept_map={
    'คลองสาน': 4,
'คลองสามวา': 2,
'คลองเตย': 5,
'คันนายาว': 3,
'จตุจักร': 38,
'จอมทอง': 4,
'ดอนเมือง': 3,
'ดินแดง': 8,
'ดุสิต': 9,
'ตลิ่งชัน': 2,
'ทวีวัฒนา': 2,
'ทุ่งครุ': 3,
'ธนบุรี': 8,
'บางกอกน้อย': 4,
'บางกอกใหญ่': 5,
'บางกะปิ': 4,
'บางขุนเทียน': 4,
'บางคอแหลม': 2,
'บางซื่อ': 5,
'บางนา': 2,
'บางบอน': 3,
'บางพลัด': 7,
'บางรัก': 4,
'บางเขน': 3,
'บางแค': 5,
'บึงกุ่ม': 3,
'ปทุมวัน': 9,
'ประเวศ': 4,
'ป้อมปราบศัตรูพ่าย': 9,
'พญาไท': 9,
'พระนคร': 16,
'พระโขนง': 1,
'ภาษีเจริญ': 3,
'มีนบุรี': 3,
'ยานนาวา': 3,
'ราชเทวี': 10,
'ราษฎร์บูรณะ': 3,
'ลาดกระบัง': 6,
'ลาดพร้าว': 2,
'วังทองหลาง': 3,
'วัฒนา': 4,
'สวนหลวง': 2,
'สะพานสูง': 1,
'สัมพันธวงศ์': 3,
'สาทร': 4,
'สายไหม': 2,
'หนองจอก': 6,
'หนองแขม': 3,
'หลักสี่': 8,
'ห้วยขวาง': 6,
}
# ---------------- Type & District Maps ----------------
type_clean_map = {
    "ถนน": 0,
    "ท่อระบายน้ำ": 1,
    "จราจร": 2,
    "ทางเท้า": 3,
    "แสงสว่าง": 4,
    "Unknown": 5,
    "ความสะอาด": 6,
    "คลอง": 7,
    "สัตว์จรจัด": 8,
    "ร้องเรียน": 9,
    "เสนอแนะ": 10,
    "กีดขวาง": 11,
    "ความปลอดภัย": 12,
    "ต้นไม้": 13,
    "เสียงรบกวน": 14,
    "การเดินทาง": 15,
    "ป้าย": 16,
    "น้ำท่วม": 17,
    "คนจรจัด": 18,
    "สะพาน": 19,
    "สายไฟ": 20,
    "ป้ายจราจร": 21,
    "สอบถาม": 22,
    "ห้องน้ำ": 23,
    "PM2.5": 24,
}
# list ของ type สำหรับ UI + one-hot (เอา Unknown ด้วย / ถ้าไม่อยากใช้ก็กรองออกได้)
TYPE_VALUES = list(type_clean_map.keys())

# map: type value -> ชื่อคอลัมน์ one-hot ที่ใช้ในโมเดล
# (ตรงกับตอน training: 'type_' + value แล้ว replace '.' เป็น '_')
TYPE_COLUMN_NAMES = {t: f"type_{t}".replace(".", "_") for t in TYPE_VALUES}

district_map = {
    "ราษฎร์บูรณะ": 0,
    "พระโขนง": 1,
    "ตลิ่งชัน": 2,
    "จตุจักร": 3,
    "มีนบุรี": 4,
    "ป้อมปราบศัตรูพ่าย": 5,
    "สาทร": 6,
    "Unknown": 7,
    "ยานนาวา": 8,
    "ปทุมวัน": 9,
    "บางแค": 10,
    "ลาดพร้าว": 11,
    "บางซื่อ": 12,
    "หนองจอก": 13,
    "ทุ่งครุ": 14,
    "บึงกุ่ม": 15,
    "คลองสามวา": 16,
    "บางขุนเทียน": 17,
    "คลองเตย": 18,
    "พระนคร": 19,
    "หนองแขม": 20,
    "ดินแดง": 21,
    "บางบอน": 22,
    "ราชเทวี": 23,
    "บางรัก": 24,
    "ดอนเมือง": 25,
    "ทวีวัฒนา": 26,
    "ธนบุรี": 27,
    "สะพานสูง": 28,
    "บางพลัด": 29,
    "สายไหม": 30,
    "ดุสิต": 31,
    "คันนายาว": 32,
    "วัฒนา": 33,
    "ประเวศ": 34,
    "พญาไท": 35,
    "บางกะปิ": 36,
    "บางคอแหลม": 37,
    "คลองสาน": 38,
    "ห้วยขวาง": 39,
    "บางเขน": 40,
    "บางกอกใหญ่": 41,
    "หลักสี่": 42,
    "วังทองหลาง": 43,
    "บางกอกน้อย": 44,
    "บางนา": 45,
    "ภาษีเจริญ": 46,
    "ลาดกระบัง": 47,
    "สวนหลวง": 48,
    "จอมทอง": 49,
    "บางกรวย": 50,
    "สัมพันธวงศ์": 51,
    "บางเสาธง": 52,
    "บางน้ำเปรี้ยว": 53,
    "เมืองสมุทรปราการ": 54,
    "พุทธมณฑล": 55,
    "พรหมพิราม": 56,
    "เมืองภูเก็ต": 57,
    "ปากเกร็ด": 58,
    "พระประแดง": 59,
    "เมืองนครสวรรค์": 60,
    "เมืองราชบุรี": 61,
    "ลำลูกกา": 62,
    "เมืองสมุทรสาคร": 63,
    "เมืองนครราชสีมา": 64,
    "กะทู้": 65,
    "บางบัวทอง": 66,
    "เมืองนนทบุรี": 67,
    "เมืองเชียงใหม่": 68,
    "เมืองปทุมธานี": 69,
    "บางพลี": 70,
    "สามพราน": 71,
    "บ้านโป่ง": 72,
    "ไทรน้อย": 73,
    "หาดใหญ่": 74,
    "สันกำแพง": 75,
}

# ---------------- Bangkok districts ----------------
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

# ---------------- Caching heavy stuff ----------------
@st.cache_resource(show_spinner=False)
def get_spark_session():
    return (
        SparkSession.builder
        .appName("TraffyFonduePredictionApp")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

@st.cache_resource(show_spinner=False)
def load_model():
    candidates = [
        "pipelines/models/gbt_spark_model"
    ]
    for path in candidates:
        if os.path.exists(path):
            return PipelineModel.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_viz_data():
    data_path = "data/processed/bangkok_traffy_merged.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# ---------------- Session state helpers ----------------
if "district" not in st.session_state:
    st.session_state.district = list(BANGKOK_DISTRICTS.keys())[0]
if "lat" not in st.session_state or "lon" not in st.session_state:
    d0 = st.session_state.district
    st.session_state.lat, st.session_state.lon = BANGKOK_DISTRICTS[d0]

def on_district_change():
    d = st.session_state.district
    lat, lon = BANGKOK_DISTRICTS[d]
    st.session_state.lat = lat
    st.session_state.lon = lon

# ---------------- Layout ----------------
st.title("🏙️ Bangkok Traffy Fondue Analytics")

tab_viz, tab_pred = st.tabs(["📊 Data Visualization", "🔮 Resolution Time Prediction"])

# ---------- Tab 1: Visualization ----------
with tab_viz:
    st.header("Bangkok Complaints Overview")
    df = load_viz_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Complaints", f"{len(df):,}")
        col2.metric("Districts", f"{df['district'].nunique()}")
        if "resolution_time" in df.columns:
            col3.metric("Avg Resolution (Hrs)", f"{df['resolution_time'].mean():.1f}")
        else:
            col3.metric("Avg Resolution (Hrs)", "N/A")
        col4.metric("Completed Tasks", f"{len(df[df['state'] == 'เสร็จสิ้น']):,}")
        
        st.divider()

        # ----- Heatmap with sampling -----
        st.subheader("📍 Complaints Heatmap")
        map_df = df.dropna(subset=["lat", "lon"])

        if not map_df.empty:
            map_df = map_df[["lon", "lat"]].copy()

            MAX_POINTS = 20_000
            if len(map_df) > MAX_POINTS:
                map_df = map_df.sample(MAX_POINTS, random_state=42)

            layer = pdk.Layer(
                "HexagonLayer",
                map_df,
                get_position="[lon, lat]",
                auto_highlight=True,
                elevation_scale=50,
                pickable=True,
                elevation_range=[0, 3000],
                extruded=True,
                coverage=1,
            )
            view_state = pdk.ViewState(
                latitude=13.7563,
                longitude=100.5018,
                zoom=10,
                pitch=50,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "Count: {elevationValue}"},
                )
            )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Complaints by Type")
            if "type_clean" in df.columns:
                type_counts = df["type_clean"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig = px.bar(
                    type_counts,
                    x="Type",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 10 Districts with Most Complaints")
            if "district" in df.columns:
                dist_counts = df["district"].value_counts().head(10).reset_index()
                dist_counts.columns = ["District", "Count"]
                fig = px.bar(
                    dist_counts,
                    x="Count",
                    y="District",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Magma",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data not found. Please run the data pipeline to generate the dataset.")

# ---------- Tab 2: Prediction ----------
with tab_pred:
    st.header("🔮 Predict Resolution Time")
    st.markdown("Fill in the details below to predict how long it will take to resolve a complaint.")

    spark = get_spark_session()
    model = load_model()

    if model is None:
        st.error("Model not found. Please train & save the model first.")
    else:
        district = st.selectbox(
            "Select District",
            list(BANGKOK_DISTRICTS.keys()),
            key="district",
            on_change=on_district_change,
        )
        default_lat = st.session_state.lat
        default_lon = st.session_state.lon

        with st.form("prediction_form"):
            st.subheader("📝 Complaint Details")

            col1, col2 = st.columns(2)

            with col1:
                organization = st.text_input(
                    "Organization",
                    value=f"เขต{district}",
                )
                # ✅ ให้เลือก type ครบทุกค่าใน type_clean_map
                type_clean = st.selectbox(
                    "Type",
                    TYPE_VALUES,
                    index=TYPE_VALUES.index("ถนน") if "ถนน" in TYPE_VALUES else 0,
                )
                timestamp = st.date_input(
                    "Date", value=pd.to_datetime("today")
                )
                hour = st.slider("Hour of Day", 0, 23, 12)

            with col2:
                lat = st.number_input(
                    "Latitude",
                    key="lat",
                    value=float(default_lat),
                    format="%.4f",
                )
                lon = st.number_input(
                    "Longitude",
                    key="lon",
                    value=float(default_lon),
                    format="%.4f",
                )
                comment_len = st.number_input(
                    "Comment Length (chars)", value=50
                )
                count_reopen = st.number_input("Reopen Count", value=0)

            st.subheader("🌤️ Environmental Factors")
            col3, col4 = st.columns(2)

            with col3:
                rainfall_mm = st.number_input("Rainfall (mm)", value=0.0)
                has_rain = st.checkbox("Is Raining?", value=False)

            with col4:
                pm25_avg = st.number_input("PM2.5 Level", value=20.0)
                pm10_avg = st.number_input("PM10 Level", value=40.0)
                dust_avg = st.number_input("Dust Level", value=30.0)

            submitted = st.form_submit_button(
                "🚀 Predict Resolution Time", type="primary"
            )

        if submitted:
            from pyspark.sql.functions import col as spark_col

            # Prepare timestamp string
            timestamp_str = f"{timestamp} {hour:02d}:00:00"

            # district_code ตรงกับที่ใช้ใน training
            district_code = district_map.get(district, district_map["Unknown"])

            # ✅ dept_count ตาม mapping ของเขต (ถ้าไม่เจอให้เป็น 0.0)
            dept_val = float(dept_map.get(district, 0.0))

            # ---------- สร้าง dict base features ----------
            data = {
                # cols ที่ใช้ใน training modeler.base_feature_cols
                "coords": [f"{lon},{lat}"],  # จะโดน cast เป็น double แล้ว fillna(0)
                "timestamp": [timestamp_str],
                "count_reopen": [float(count_reopen)],
                "comment_len": [float(comment_len)],
                "lon": [float(lon)],
                "lat": [float(lat)],
                "pm25_avg": [float(pm25_avg)],
                "pm10_avg": [float(pm10_avg)],
                "dust_avg": [float(dust_avg)],
                "rainfall_mm": [float(rainfall_mm)],
                "has_rain": [1.0 if has_rain else 0.0],
                "dept_count": [dept_val],              # 👈 ใช้ dept_map แล้ว
                "district_code": [float(district_code)],
                # cols extra ที่ไม่ใช้ แต่เก็บไว้เฉย ๆ ก็ไม่เป็นไร
                "organization": [organization],
                "type_clean": [type_clean],
                "district": [district],
            }

            # ---------- ใส่ one-hot type_* ให้ครบทุกตัว ----------
            for t in TYPE_VALUES:
                col_name = TYPE_COLUMN_NAMES[t]  # เช่น 'type_PM2_5'
                data[col_name] = [1.0 if t == type_clean else 0.0]

            # สร้าง Spark DataFrame
            pdf = pd.DataFrame(data)
            input_df = spark.createDataFrame(pdf)

            # cast features เป็น double ให้ชัวร์
            feature_cols_for_cast = [
                "coords",
                "timestamp",
                "count_reopen",
                "comment_len",
                "lon",
                "lat",
                "pm25_avg",
                "pm10_avg",
                "dust_avg",
                "rainfall_mm",
                "has_rain",
                "dept_count",
                "district_code",
            ] + list(TYPE_COLUMN_NAMES.values())

            for c in feature_cols_for_cast:
                if c in input_df.columns:
                    input_df = input_df.withColumn(c, spark_col(c).cast("double"))

            # เติม 0 สำหรับ missing (เผื่ออนาคต)
            input_df = input_df.fillna(0)

            try:
                prediction = model.transform(input_df)
                log_pred = prediction.select("prediction").collect()[0][0]
                hours_pred = float(np.expm1(log_pred))

                st.success(f"⏱️ Predicted Resolution Time: **{hours_pred:.2f} hours**")
                if hours_pred > 24:
                    st.info(
                        f"📅 That's approximately **{hours_pred/24:.1f} days**."
                    )

                with st.expander("Debug input row"):
                    st.write(pdf)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                with st.expander("Debug Info"):
                    st.write(data)

