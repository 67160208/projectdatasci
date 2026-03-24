import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. การตั้งค่าหน้าเว็บ (Page Configuration) ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# --- 2. โหลดโมเดล (Cache ไว้จะได้ไม่โหลดใหม่ทุกครั้งที่ขยับ UI) ---
@st.cache_resource
def load_model():
    # Model file names เป็น joblib ตามปกติ
    candidates = [
        os.path.join(os.path.dirname(__file__), 'salary_rf_pipeline.joblib'),
        'salary_rf_pipeline.joblib',
        '/mount/src/projectdatasci/salary_rf_pipeline.joblib',
    ]

    # 1) ถ้ามีอยู่ local ก็โหลดทันที
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)

    # 2) ถ้าไม่มี local ให้ลองดาวน์โหลดจาก URL ที่กำหนดไว้
    #    ถ้าไม่กำหนด ENV ใช้ Google Drive direct download ที่ให้ไว้เป็น Default
    default_model_url = "https://drive.google.com/uc?export=download&id=13ph4-aSXnTgJM9KP2G9Ra81gPU5pRUON"
    model_url = os.environ.get('MODEL_URL', default_model_url)

    local_name = os.path.join(os.path.dirname(__file__), os.path.basename(model_url))
    try:
        import urllib.request
        st.info(f"ดาวน์โหลดโมเดลจาก URL: {model_url} ไปยัง {local_name}")
        urllib.request.urlretrieve(model_url, local_name)
        return joblib.load(local_name)
    except Exception as e:
        raise RuntimeError(f"โหลดโมเดลจาก MODEL_URL ไม่สำเร็จ: {e}\nURL ที่ใช้: {model_url}")

    # 3) ถ้ายังไม่เจอ ให้แจ้งให้ชัด
    checked = ', '.join(candidates)
    raise FileNotFoundError(
        "ไม่พบไฟล์โมเดล salary_rf_pipeline.joblib ใน path เหล่านี้: "
        + checked + "\nกรุณาตั้งค่า MODEL_URL=... (S3/Drive) และรีสตาร์ท หรือวางไฟล์ไว้ใน folder เดียวกับ app.py"
    )

model = load_model()

# --- Dictionary สำหรับแปลง Education เป็น Education-Num อัตโนมัติ ---
edu_num_map = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, 
    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 
    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 
    'Prof-school': 15, 'Doctorate': 16
}

# --- 3. ส่วนหัวของเว็บ (Header & Disclaimer) ---
st.title("💰 ระบบทำนายช่วงรายได้ (Salary Prediction)")
st.markdown("""
แอปพลิเคชันนี้ใช้โมเดล Machine Learning (Random Forest) ในการทำนายว่าบุคคลหนึ่งๆ จะมีรายได้ **มากกว่า $50,000 ต่อปี** หรือไม่ โดยอ้างอิงจากข้อมูลประชากรศาสตร์และการทำงาน
""")

st.info("**Disclaimer:** แอปพลิเคชันนี้เป็นส่วนหนึ่งของโครงงานวิชาเรียน (Student Project) ข้อมูลและการทำนายถูกสร้างขึ้นเพื่อจุดประสงค์ทางการศึกษาเท่านั้น ไม่ควรนำไปใช้ตัดสินใจทางการเงินจริง")

# --- 4. ส่วนรับข้อมูลจากผู้ใช้ (User Inputs) ---
st.header("📋 กรอกข้อมูลของคุณ")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ข้อมูลส่วนตัว")
    # Input Validation: กำหนดช่วงอายุที่สมเหตุสมผล 17-90 ปี
    age = st.number_input("อายุ (Age)", min_value=17, max_value=90, value=30, step=1)
    sex = st.selectbox("เพศ (Sex)", ['Male', 'Female'])
    race = st.selectbox("เชื้อชาติ (Race)", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    native_country = st.selectbox("ประเทศบ้านเกิด (Native Country)", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Honduras', 'Hungary', 'Scotland', 'Holand-Netherlands'])
    marital_status = st.selectbox("สถานภาพสมรส (Marital Status)", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    relationship = st.selectbox("สถานะในครอบครัว (Relationship)", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])

with col2:
    st.subheader("ข้อมูลการศึกษาและการทำงาน")
    education = st.selectbox("ระดับการศึกษา (Education)", list(edu_num_map.keys()), index=12) # Default ที่ Bachelors
    workclass = st.selectbox("ประเภทการจ้างงาน (Workclass)", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked'])
    occupation = st.selectbox("อาชีพ (Occupation)", ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
    
    # Input Validation: กำหนดชั่วโมงทำงาน 1-99 ชั่วโมง
    hours_per_week = st.number_input("ชั่วโมงทำงานต่อสัปดาห์ (Hours per week)", min_value=1, max_value=99, value=40, step=1)
    
    # Input Validation: กำหนดให้ไม่ติดลบ
    capital_gain = st.number_input("รายได้จากสินทรัพย์ (Capital Gain $)", min_value=0, value=0, step=100)
    capital_loss = st.number_input("ขาดทุนจากสินทรัพย์ (Capital Loss $)", min_value=0, value=0, step=100)

# ค่า fnlwgt เป็นค่าที่คนทั่วไปไม่ทราบ (น้ำหนักทางสถิติของสำมะโนประชากร) เราจึงเซ็ตเป็นค่า Median ทั่วไปของ Dataset ไว้หลังบ้าน
hidden_fnlwgt = 189778 

# --- 5. การทำนายผล (Prediction) ---
st.markdown("---")
if st.button("🚀 ทำนายรายได้ (Predict)", use_container_width=True):
    
    # สร้าง DataFrame ในรูปแบบเดียวกับที่ใช้ Train โมเดลเป๊ะๆ
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [hidden_fnlwgt],
        'education': [education],
        'education-num': [edu_num_map[education]],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    
    # ทำนาย Class และ ความน่าจะเป็น (Probability)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.subheader("🎯 ผลการทำนาย")
    
    if prediction == 1:
        st.success(f"### 🎉 มีแนวโน้มรายได้ **มากกว่า $50,000** ต่อปี (>50K)")
        st.write(f"**ความมั่นใจของโมเดล (Confidence):** {prediction_proba[1]*100:.2f}%")
        st.progress(float(prediction_proba[1]))
    else:
        st.warning(f"### 💵 มีแนวโน้มรายได้ **น้อยกว่าหรือเท่ากับ $50,000** ต่อปี (<=50K)")
        st.write(f"**ความมั่นใจของโมเดล (Confidence):** {prediction_proba[0]*100:.2f}%")
        st.progress(float(prediction_proba[0]))

# --- 6. คำอธิบาย Features (Feature Descriptions) ---
st.markdown("---")
with st.expander("📚 คำอธิบายตัวแปร (Feature Descriptions)"):
    st.markdown("""
    * **Age:** อายุของบุคคล (ปี)
    * **Workclass:** ประเภทของนายจ้างหรือการจ้างงาน เช่น เอกชน (Private), รัฐบาล (Gov) หรือ ธุรกิจส่วนตัว (Self-emp)
    * **Education:** ระดับการศึกษาสูงสุดที่ได้รับ
    * **Marital Status:** สถานภาพทางการสมรส
    * **Occupation:** สายอาชีพหรือลักษณะงานที่ทำ
    * **Relationship:** สถานะความสัมพันธ์ในครอบครัว เช่น เป็นหัวหน้าครอบครัว (Husband/Wife) หรืออยู่คนเดียว (Not-in-family)
    * **Capital Gain / Loss:** รายได้หรือผลขาดทุนที่เกิดจากการขายสินทรัพย์ (เช่น หุ้น หรือ อสังหาริมทรัพย์) นอกเหนือจากเงินเดือน
    * **Hours per week:** จำนวนชั่วโมงที่ทำงานในหนึ่งสัปดาห์
    """)
