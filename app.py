import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰", layout="centered")
st.title("💰 ระบบทำนายรายได้ประชากร")
st.write("ทำนายผลด้วยระบบ Random Forest Pipeline (รวม Scaling & Encoding)")

# --- 2. โหลด Pipeline ---
@st.cache_resource
def load_pipeline():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_pipeline()

# --- 3. ส่วนรับข้อมูลจากผู้ใช้ ---
if pipeline is not None:
    st.subheader("📊 กรอกข้อมูลเพื่อวิเคราะห์")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("อายุ (Age)", 17, 90, 45)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Federal-gov'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'Masters', 'Doctorate', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', '1st-4th', '10th', '5th-6th', 'Preschool'])
        sex = st.selectbox("เพศ", ['Male', 'Female'])
        cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 10000)

    with col2:
        marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Handlers-cleaners', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
        hours = st.slider("ชั่วโมงทำงานต่อสัปดาห์", 1, 99, 45)
        country = st.selectbox("ประเทศ", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Thailand', 'Other'])

    if st.button("🔍 วิเคราะห์รายได้", use_container_width=True):
        # 🚩 ต้องสร้าง DataFrame ให้มี 14 คอลัมน์ "ชื่อเป๊ะๆ" เหมือนในไฟล์ CSV ต้นฉบับ
        edu_num_map = {'Bachelors': 13, 'Masters': 14, 'Doctorate': 16, 'HS-grad': 9, 'Some-college': 10}
        
        raw_data = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'fnlwgt': 189778,
            'education': education,
            'education-num': edu_num_map.get(education, 10),
            'marital-status': marital,
            'occupation': occupation,
            'relationship': relationship,
            'race': 'White',
            'sex': sex,
            'capital-gain': cap_gain,
            'capital-loss': 0,
            'hours-per-week': hours,
            'native-country': country
        }])

        try:
            # 🎯 ให้ Pipeline จัดการแปลงร่างข้อมูล (Scaling & One-Hot) ให้เองอัตโนมัติ
            prediction = pipeline.predict(raw_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
                st.balloons()
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.error("❌ ไม่สามารถโหลดไฟล์ salary_pipeline.pkl ได้")
