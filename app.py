import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰", layout="centered")
st.title("💰 ระบบทำนายรายได้ประชากร")
st.write("โมเดล Random Forest Pipeline (รวมเครื่องมือแปลงข้อมูลอัตโนมัติ)")

# --- 2. โหลดโมเดล Pipeline (ตัวเดียวจบ) ---
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
    with st.form("my_form"):
        st.subheader("📊 กรอกข้อมูลส่วนบุคคล")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("อายุ (Age)", 17, 90, 45)
            workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay'])
            education = st.selectbox("การศึกษา", ['Bachelors', 'Masters', 'Doctorate', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', '1st-4th', '10th', '5th-6th', 'Preschool'])
            sex = st.selectbox("เพศ", ['Male', 'Female'])
            cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 10000)

        with col2:
            marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
            occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Handlers-cleaners', 'Other-service', 'Priv-house-serv'])
            relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
            hours = st.slider("ชั่วโมงทำงานต่อสัปดาห์", 1, 99, 45)
            country = st.selectbox("ประเทศเกิด", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Thailand', 'Other'])

        submit = st.form_submit_button("🔍 วิเคราะห์รายได้", use_container_width=True)

    if submit:
        # Mapping ระดับการศึกษาเป็นตัวเลข (ตรงตามไฟล์ Notebook ของพี่)
        edu_map = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
        
        # 🚩 หัวใจสำคัญ: ส่งข้อมูล 14 คอลัมน์ดั้งเดิม ห้ามใช้ชื่อขีดล่าง (_) ในบางตัว
        raw_data = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'fnlwgt': 200000, 
            'education': education,
            'education-num': edu_map.get(education, 10),
            'marital-status': marital,  # ใช้ขีดกลาง (-) ตามไฟล์ Notebook
            'occupation': occupation,
            'relationship': relationship,
            'race': 'White',
            'sex': sex,
            'capital-gain': cap_gain,
            'capital-loss': 0,
            'hours-per-week': hours,
            'native-country': country  # ใช้ขีดกลาง (-) ตามไฟล์ Notebook
        }])

        try:
            # 🎯 Pipeline จะจัดการเรื่อง Scaling และ Encoding ให้เองอัตโนมัติ
            prediction = pipeline.predict(raw_data)[0]
            # แอบดูความน่าจะเป็น (Debug)
            prob = pipeline.predict_proba(raw_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี** (ความมั่นใจ {prob[1]*100:.2f}%)")
                st.balloons()
            else:
                st.info(f"📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี** (ความมั่นใจ {prob[0]*100:.2f}%)")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.error("❌ หาไฟล์ salary_pipeline.pkl ไม่เจอ")
