import streamlit as st
import pandas as pd
import joblib
import os

# --- นำเข้าเครื่องมือที่จำเป็นสำหรับอ่านโมเดล (สำคัญมาก!) ---
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

# --- โหลดโมเดล ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    
    if os.path.exists(model_path):
        # ใช้การโหลดแบบระบุว่าถ้าหาคลาสไม่เจอให้ลองเช็คใน sklearn
        return joblib.load(model_path)
    else:
        st.error(f"❌ หาไฟล์โมเดลไม่เจอที่: {model_path}")
        return None

model = load_model()

# --- ส่วนรับข้อมูลและทำนายผล (เหมือนเดิม) ---
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

    if st.button("ทำนายผลรายได้", use_container_width=True):
        input_data = pd.DataFrame([[age, workclass, education, marital_status, occupation, hours]], 
                                 columns=['age', 'workclass', 'education', 'marital-status', 'occupation', 'hours-per-week'])
        res = model.predict(input_data)[0]
        if res == 1:
            st.success("🎉 รายได้น่าจะมากกว่า $50,000 ต่อปี")
        else:
            st.info("📊 รายได้น่าจะน้อยกว่าหรือเท่ากับ $50,000 ต่อปี")
