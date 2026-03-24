import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Salary Predictor", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ (ฉบับยกเครื่องใหม่)")

# โหลดโรงงานอัจฉริยะ
@st.cache_resource
def load_model():
    return joblib.load('salary_pipeline.pkl')

model = load_model()

if model:
    with st.form("main_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("อายุ", 17, 90, 45)
            workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-inc', 'Local-gov', 'State-gov'])
            edu = st.selectbox("การศึกษา", ['Bachelors', 'Masters', 'Doctorate', 'HS-grad'])
            cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 15000)
        with col2:
            marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced'])
            occ = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Sales'])
            hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 45)
            sex = st.selectbox("เพศ", ['Male', 'Female'])
        
        if st.form_submit_button("🔍 วิเคราะห์ผล"):
            # 🚩 สร้างข้อมูลดิบ 14 คอลัมน์ส่งเข้าโรงงาน
            data = pd.DataFrame([{
                'age': age, 'workclass': workclass, 'fnlwgt': 200000,
                'education': edu, 'education-num': 13, # ตัวเลขสมมติ
                'marital-status': marital, 'occupation': occ,
                'relationship': 'Husband' if sex == 'Male' else 'Wife',
                'race': 'White', 'sex': sex,
                'capital-gain': cap_gain, 'capital-loss': 0,
                'hours-per-week': hours, 'native-country': 'United-States'
            }])
            
            # ให้ Pipeline จัดการ Scaling และ Encoding เอง!
            result = model.predict(data)[0]
            
            if result == 1:
                st.success("🎉 รายได้มากกว่า $50,000")
                st.balloons()
            else:
                st.info("📊 รายได้น้อยกว่าหรือเท่ากับ $50,000")
