import os
import joblib
import streamlit as st

# 1. บังคับหาตำแหน่งโฟลเดอร์ปัจจุบันที่ไฟล์ app.py อยู่
curr_path = os.path.dirname(__file__)
model_path = os.path.join(curr_path, 'model_artifacts', 'salary_pipeline.pkl')

# 2. ฟังก์ชันโหลดโมเดลแบบเช็คชัวร์
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # ถ้าหาไม่เจอ ให้โชว์รายชื่อไฟล์ที่มีอยู่เลย จะได้รู้ว่ามันไปแอบที่ไหน
        st.error(f"❌ หาไฟล์ไม่เจอที่: {model_path}")
        st.write("ไฟล์ที่มีในโฟลเดอร์ตอนนี้คือ:", os.listdir(curr_path))
        return None

model = load_model()

# เช็คว่าโหลดติดไหมก่อนไปต่อ
if model is None:
    st.stop()
