import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

# --- 2. โหลดโมเดล ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- 3. ส่วนรับข้อมูลจากผู้ใช้ ---
if model is not None:
    st.subheader("กรอกข้อมูลเพื่อทำนายรายได้")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ", min_value=17, max_value=90, value=30)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    with col2:
        marital_status = st.selectbox("สถานภาพ", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("อาชีพ", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 40)

    st.markdown("---")
    if st.button("ทำนายผลรายได้", use_container_width=True):
        # 🚩 จุดสำคัญ: ชื่อคอลัมน์ต้องตรงกับ metadata.json เป๊ะๆ (สังเกตเครื่องหมาย - )
        data_dict = {
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [189778],
            'education': [education],
            'education-num': [10],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': ['Husband'],
            'race': ['White'],
            'sex': ['Male'],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [hours],
            'native-country': ['United-States']
        }
        
        input_data = pd.DataFrame(data_dict)
        
        # เรียงลำดับคอลัมน์ให้ตรงตามที่โมเดลต้องการ (14 columns)
        feature_order = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ]
        input_data = input_data[feature_order]

        try:
            # ทำนาย
            res = model.predict(input_data)[0]
            if res == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
            # แสดงชื่อคอลัมน์ที่โมเดลต้องการออกมาดูเพื่อเช็ค
            if hasattr(model, 'feature_names_in_'):
                st.write("โมเดลต้องการคอลัมน์ชื่อ:", list(model.feature_names_in_))
else:
    st.error("❌ ไม่สามารถโหลดโมเดลได้")
