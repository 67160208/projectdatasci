import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Adult Income Prediction", page_icon="💰")
st.title("💰 ระบบทำนายรายได้ประชากร")

@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'salary_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is not None:
    st.subheader("กรอกข้อมูลเพื่อทำนายรายได้")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ", min_value=17, max_value=90, value=30)
        workclass = st.selectbox("ประเภทงาน", ['Private', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', 'Federal-gov'])
        education = st.selectbox("การศึกษา", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    with col2:
        marital_status = st.selectbox("สถานภาพ", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("อาชีพ", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        hours = st.slider("ชั่วโมงทำงาน/สัปดาห์", 1, 99, 40)

    if st.button("ทำนายผลรายได้", use_container_width=True):
        # 1. รายชื่อคอลัมน์ทั้ง 96 อันที่โมเดลต้องการ (จาก Error Log ของพี่)
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

        # 2. สร้าง DataFrame เริ่มต้นเป็น 0 ทั้งหมด
        input_df = pd.DataFrame(0, index=[0], columns=all_features)

        # 3. เติมข้อมูลตัวเลข
        input_df['age'] = age
        input_df['fnlwgt'] = 189778 # ค่ากลาง
        input_df['education-num'] = 10 # ค่ากลาง
        input_df['capital-gain'] = 0
        input_df['capital-loss'] = 0
        input_df['hours-per-week'] = hours

        # 4. เติมข้อมูลหมวดหมู่ (เปลี่ยน 0 เป็น 1 ในช่องที่เลือก)
        for col_name, value in [('workclass', workclass), ('education', education), 
                                ('marital-status', marital_status), ('occupation', occupation)]:
            target_col = f"{col_name}_{value}"
            if target_col in all_features:
                input_df[target_col] = 1

        # ตั้งค่าค่าคงที่อื่นๆ (เช่น race, sex, native-country) ให้เป็นค่ามาตรฐาน
        input_df['race_White'] = 1
        input_df['sex_Male'] = 1
        input_df['native-country_United-States'] = 1

        # 5. ทำนาย
        try:
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี**")
            else:
                st.info("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี**")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
