import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Adult Income Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

@st.cache_resource
def load_pipeline():
    base_path = os.path.dirname(__file__)
    # ต้องมั่นใจว่าเป็นไฟล์ที่ได้จาก Step 1 นะพี่!
    return joblib.load(os.path.join(base_path, 'salary_pipeline.pkl'))

pipeline = load_pipeline()

if pipeline is not None:
    st.subheader("📊 กรอกข้อมูลส่วนบุคคล")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ", 17, 90, 45)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-inc', 'Local-gov', 'State-gov', 'Federal-gov'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'Masters', 'Doctorate', 'HS-grad', 'Some-college'])
        sex = st.selectbox("เพศ", ['Male', 'Female'])
        cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 15000) # ลองใส่ 15000
    with col2:
        marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced'])
        relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried'])
        occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Sales', 'Craft-repair'])
        hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 45)
        country = st.selectbox("ประเทศเกิด", ['United-States', 'Thailand', 'Mexico'])

    if st.button("🔍 วิเคราะห์รายได้"):
        # 🚩 ส่งข้อมูลดิบ 14 คอลัมน์ (ชื่อต้องเป๊ะตาม CSV)
        raw_data = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': 200000, 
            'education': education, 'education-num': 13,
            'marital-status': marital, 'occupation': occupation,
            'relationship': relationship, 'race': 'White', 'sex': sex,
            'capital-gain': cap_gain, 'capital-loss': 0,
            'hours-per-week': hours, 'native-country': country
        }])

        prediction = pipeline.predict(raw_data)[0]
        if prediction == 1:
            st.success("💰 ผล: **มากกว่า $50,000 ต่อปี**")
            st.balloons()
        else:
            st.info("📊 ผล: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
