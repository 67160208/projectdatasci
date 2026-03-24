import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 ระบบทำนายรายได้ประชากร")
st.markdown("---")

# --- 2. ฟังก์ชันโหลดโมเดล (แบบเช็ค Path ละเอียด) ---
@st.cache_resource
def load_model():
    # หาตำแหน่งที่ไฟล์ app.py นี้อยู่
    base_path = os.path.dirname(__file__)
    # ระบุตำแหน่งไฟล์โมเดลในโฟลเดอร์ model_artifacts
    model_path = os.path.join(base_path, 'model_artifacts', 'salary_pipeline.pkl')
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ ไม่พบไฟล์โมเดลที่ตำแหน่ง: {model_path}")
        st.info("กรุณาตรวจสอบว่าใน GitHub มีโฟลเดอร์ 'model_artifacts' และไฟล์ 'salary_pipeline.pkl' อยู่จริง")
        return None

# โหลดโมเดล
model = load_model()

# --- 3. ส่วนรับข้อมูลจากผู้ใช้ (UI) ---
if model is not None:
    st.subheader("กรอกข้อมูลส่วนบุคคล")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("อายุ (Age)", min_value=17, max_value=90, value=30)
        workclass = st.selectbox("ประเภทการจ้างงาน (Workclass)", 
                                ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 
                                 'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
        education = st.selectbox("ระดับการศึกษา (Education)", 
                                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 
                                 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        
    with col2:
        marital_status = st.selectbox("สถานภาพสมรส (Marital Status)", 
                                    ['Never-married', 'Married-civ-spouse', 'Divorced', 
                                     'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("อาชีพ (Occupation)", 
                                 ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 
                                  'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 
                                  'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 
                                  'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        hours_per_week = st.slider("ชั่วโมงทำงานต่อสัปดาห์", 1, 99, 40)

    # --- 4. การทำนายผล ---
    st.markdown("---")
    if st.button("ประมวลผลทำนายรายได้", use_container_width=True):
        # สร้าง DataFrame ให้เหมือนกับตอนที่ใช้เทรนโมเดล
        input_df = pd.DataFrame([[age, workclass, education, marital_status, occupation, hours_per_week]], 
                               columns=['age', 'workclass', 'education', 'marital-status', 'occupation', 'hours-per-week'])
        
        try:
            # ทำนายผล
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None

            # แสดงผลลัพธ์
            st.subheader("ผลการวิเคราะห์:")
            if prediction == 1:
                st.success("🎉 คาดการณ์รายได้: **มากกว่า $50,000 ต่อปี** (>50K)")
            else:
                st.warning("📊 คาดการณ์รายได้: **น้อยกว่าหรือเท่ากับ $50,000 ต่อปี** (<=50K)")
            
            # (Option) แสดงค่าความมั่นใจถ้าโมเดลรองรับ
            if probability is not None:
                st.write(f"ระดับความมั่นใจ: {max(probability[0])*100:.2f}%")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

else:
    st.warning("⚠️ เว็บไซต์ยังไม่พร้อมทำงาน เนื่องจากโหลดโมเดลไม่สำเร็จ")

# ส่วนท้าย
st.markdown("---")
st.caption("Project: Adult Income Prediction | Data Science Course")
