FROM jupyter/pyspark-notebook:latest

# ลงแค่ของที่ต้องใช้เพิ่ม (อย่าแตะ pyspark เลย ให้ base image จัดการ)
RUN pip install --no-cache-dir pyspark==3.5.0 streamlit

WORKDIR /home/jovyan/work

# Copy requirements file
COPY requirements.txt /tmp/

# ตรงนี้สำคัญ: ให้แน่ใจว่าใน requirements.txt **ไม่มี pyspark**
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888 4040 8501
