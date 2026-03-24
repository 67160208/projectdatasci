import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Adult Income Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

@st.cache_resource
def load_data():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        obj = joblib.load(model_path)
        # ตรวจสอบว่าเป็น Pipeline หรือไม่
        if not hasattr(obj, 'steps'):
            st.error("⚠️ ไฟล์ .pkl ของพี่ไม่ใช่ Pipeline! ผลคำนวณจะไม่แม่นยำ โปรดทำตามสเต็ปที่ 1 ใน Colab ใหม่ครับ")
        return obj
    return None

pipeline = load_data()

if pipeline is not None:
    st.subheader("📊 กรอกข้อมูลส่วนบุคคล")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ", 17, 90, 45)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-inc', 'Local-gov', 'State-gov', 'Federal-gov'])
        education = st.selectbox("การศึกษา", ['Masters', 'Doctorate', 'Bachelors', 'HS-grad', 'Some-college'])
        sex = st.selectbox("เพศ", ['Male', 'Female'])
        cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 20000)
    with col2:
        marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced'])
        relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried'])
        occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Sales'])
        hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 45)
        country = st.selectbox("ประเทศ", ['United-States', 'Thailand', 'Mexico'])

    if st.button("🔍 วิเคราะห์รายได้"):
        # สร้างตาราง 14 คอลัมน์ดิบๆ ส่งเข้าท่อ Pipeline
        raw_df = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': 200000, 
            'education': education, 'education-num': 14 if education == 'Masters' else 16 if education == 'Doctorate' else 13,
            'marital-status': marital, 'occupation': occupation,
            'relationship': relationship, 'race': 'White', 'sex': sex,
            'capital-gain': cap_gain, 'capital-loss': 0,
            'hours-per-week': hours, 'native-country': country
        }])

        try:
            prediction = pipeline.predict(raw_df)[0]
            st.markdown("---")
            if prediction == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
                st.balloons()
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
