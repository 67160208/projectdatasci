import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Adult Income Prediction", page_icon="💰", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 ระบบทำนายรายได้ประชากร")
st.write("โปรเจกต์วิเคราะห์ข้อมูลรายได้จากปัจจัยทางสังคมและอาชีพ")

# --- 2. ฟังก์ชันโหลดโมเดล ---
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
    with st.container():
        st.subheader("📊 กรอกข้อมูลส่วนบุคคล")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("อายุ (Age)", min_value=17, max_value=90, value=30)
            workclass = st.selectbox("ประเภทงาน (Workclass)", 
                                   ['Private', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Federal-gov'])
            education = st.selectbox("ระดับการศึกษา (Education)", 
                                   ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
            sex = st.selectbox("เพศ (Sex)", ['Male', 'Female'])
            cap_gain = st.number_input("กำไรจากทรัพย์สิน (Capital Gain)", min_value=0, max_value=99999, value=0, help="คนที่มีรายได้สูงส่วนใหญ่มักมีค่านี้มากกว่า 5,000")
        
        with col2:
            marital_status = st.selectbox("สถานภาพ (Marital Status)", 
                                        ['Married-civ-spouse', 'Never-married', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
            relationship = st.selectbox("ความสัมพันธ์ (Relationship)", 
                                      ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
            occupation = st.selectbox("กลุ่มอาชีพ (Occupation)", 
                                    ['Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Handlers-cleaners', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
            hours = st.slider("ชั่วโมงทำงานต่อสัปดาห์", 1, 99, 40)

    st.markdown("---")

    # --- 4. ส่วนคำนวณและทำนายผล ---
    if st.button("🔍 วิเคราะห์รายได้"):
        all_features = [
            'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
            'workclass_Local-gov', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay',
            'education_11th', 'education_12th', 'education_1st-4th', 'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad', 'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college',
            'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed',
            'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
            'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
            'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'sex_Male',
            'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia'
        ]

        input_df = pd.DataFrame(0, index=[0], columns=all_features)

        input_df['age'] = age
        input_df['fnlwgt'] = 189778
        input_df['capital-gain'] = cap_gain
        input_df['capital-loss'] = 0
        input_df['hours-per-week'] = hours
        
        edu_map = {
            'Bachelors': 13, 'Some-college': 10, '11th': 7, 'HS-grad': 9, 
            'Prof-school': 15, 'Assoc-acdm': 12, 'Assoc-voc': 11, '9th': 5, 
            '7th-8th': 4, '12th': 8, 'Masters': 14, '1st-4th': 2, 
            '10th': 6, 'Doctorate': 16, '5th-6th': 3, 'Preschool': 1
        }
        input_df['education-num'] = edu_map.get(education, 10)

        # เติมค่า One-Hot Encoding
        for col, val in [('workclass', workclass), ('education', education), 
                         ('marital-status', marital_status), ('occupation', occupation),
                         ('relationship', relationship)]:
            target = f"{col}_{val}"
            if target in all_features:
                input_df[target] = 1

        # เพศ (ใช้ logic sex_Male: 1=Male, 0=Female)
        if sex == 'Male':
            input_df['sex_Male'] = 1

        # ค่ามาตรฐานอื่นๆ
        input_df['race_White'] = 1
        input_df['native-country_United-States'] = 1

        try:
            prediction = model.predict(input_df)[0]
            st.subheader("🏁 ผลการวิเคราะห์:")
            if prediction == 1:
                st.success("💰 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
                st.balloons()
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

else:
    st.error("❌ ไม่สามารถเชื่อมต่อกับฐานข้อมูลโมเดลได้")

st.sidebar.info("ระบบทำนายรายได้ประชากร | Random Forest Classifier")
