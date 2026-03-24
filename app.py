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
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #FF4B4B; color: white; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 ระบบทำนายรายได้ประชากร")

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- 3. ส่วนรับข้อมูล ---
if model is not None:
    with st.container():
        st.subheader("📊 กรอกข้อมูลส่วนบุคคล")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("อายุ (Age)", 17, 90, 45)
            workclass = st.selectbox("ประเภทงาน", ['Private', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Federal-gov'])
            education = st.selectbox("การศึกษา", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
            sex = st.selectbox("เพศ", ['Male', 'Female'])
        with col2:
            marital = st.selectbox("สถานภาพ", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
            relationship = st.selectbox("ความสัมพันธ์", ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
            occupation = st.selectbox("อาชีพ", ['Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Handlers-cleaners', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
            cap_gain = st.number_input("กำไร (Capital Gain)", 0, 99999, 10000)
            hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 45)

    if st.button("🔍 วิเคราะห์รายได้"):
        # รายชื่อ 96 คอลัมน์เป๊ะๆ จาก Log ของพี่
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

        # สร้าง DataFrame 0 ล้วน
        input_df = pd.DataFrame(0, index=[0], columns=all_features)

        # เติมค่า Numerical
        input_df['age'] = age
        input_df['fnlwgt'] = 189778
        input_df['capital-gain'] = cap_gain
        input_df['hours-per-week'] = hours
        
        edu_map = {'Bachelors': 13, 'Masters': 14, 'Doctorate': 16, 'HS-grad': 9, 'Some-college': 10}
        input_df['education-num'] = edu_map.get(education, 10)

        # เติมค่า Categorical (One-Hot)
        for col_name, val in [('workclass', workclass), ('education', education), 
                                ('marital-status', marital), ('occupation', occupation),
                                ('relationship', relationship), ('race', 'White')]:
            target = f"{col_name}_{val}"
            if target in all_features:
                input_df[target] = 1

        # เพศ (Male=1, Female=0)
        if sex == 'Male':
            input_df['sex_Male'] = 1
            
        input_df['native-country_United-States'] = 1

        # 🎯 จุดเปลี่ยนชีวิต: ทำนายผล
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("💰 ผลการวิเคราะห์: **มากกว่า $50,000 ต่อปี**")
            st.balloons()
        else:
            st.info("📊 ผลการวิเคราะห์: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
            st.write("💡 ลองปรับสถานภาพเป็น Married-civ-spouse และอาชีพเป็น Exec-managerial ดูครับ")
