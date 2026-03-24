import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Salary Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

# 1. โหลด Pipeline ตัวใหม่
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 2. ส่วนรับข้อมูล
if model is not None:
    st.subheader("📊 กรอกข้อมูลเพื่อวิเคราะห์")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ", 17, 90, 45)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-inc', 'Local-gov', 'State-gov'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'Masters', 'Doctorate', 'HS-grad'])
        cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 15000)
    with col2:
        marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced'])
        occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Sales'])
        hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 45)
        sex = st.selectbox("เพศ", ['Male', 'Female'])

    if st.button("🔍 วิเคราะห์รายได้"):
        # 🚩 ส่งข้อมูลดิบ 14 คอลัมน์เข้า Pipeline
        input_df = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': 200000, 
            'education': education, 'education-num': 13,
            'marital-status': marital, 'occupation': occupation,
            'relationship': 'Husband' if sex == 'Male' else 'Wife',
            'race': 'White', 'sex': sex,
            'capital-gain': cap_gain, 'capital-loss': 0,
            'hours-per-week': hours, 'native-country': 'United-States'
        }])

        # Pipeline จะทำ Scaling และ One-Hot ให้เองอัตโนมัติ!
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("🎉 ผลการทำนาย: **มากกว่า $50,000 ต่อปี**")
            st.balloons()
        else:
            st.info("📊 ผลการทำนาย: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
