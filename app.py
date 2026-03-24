import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰", layout="centered")

st.title("💰 ระบบทำนายรายได้ประชากร")
st.write("ทายผลรายได้ด้วยโมเดล Random Forest Pipeline (รวม Scaling & Encoding)")

# --- 2. โหลด Pipeline (มัดรวมทุกอย่างไว้แล้ว) ---
@st.cache_resource
def load_pipeline():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_pipeline()

# --- 3. ส่วนรับข้อมูลจากผู้ใช้ (รับเป็นค่าดิบๆ) ---
if pipeline is not None:
    st.subheader("📊 กรอกข้อมูลเพื่อวิเคราะห์")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("อายุ (Age)", 17, 90, 30)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Federal-gov'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        sex = st.selectbox("เพศ", ['Male', 'Female'])
        cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 0)

    with col2:
        marital = st.selectbox("สถานภาพ", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("อาชีพ", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
        hours = st.slider("ชั่วโมงทำงานต่อสัปดาห์", 1, 99, 40)
        country = st.selectbox("ประเทศ", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Thailand', 'Other'])

    if st.button("🔍 วิเคราะห์รายได้", use_container_width=True):
        # 🚩 หัวใจสำคัญ: ส่งข้อมูลดิบ 14 คอลัมน์ให้ตรงตามที่ Pipeline ต้องการ
        raw_data = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'fnlwgt': 189778,
            'education': education,
            'education-num': 10, # เดี๋ยว Pipeline จะไปจัดการเอง หรือพี่จะทำ Map มาใส่ก็ได้
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
            # Pipeline จะจัดการ Scaling (ลดขนาดตัวเลข) และ One-Hot ให้เองอัตโนมัติ!
            prediction = pipeline.predict(raw_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
                st.balloons()
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.write("ลองตรวจสอบชื่อคอลัมน์ใน DataFrame ให้ตรงกับตอนเทรนนะครับ")
else:
    st.error("❌ หาไฟล์ salary_pipeline.pkl ไม่เจอ")
