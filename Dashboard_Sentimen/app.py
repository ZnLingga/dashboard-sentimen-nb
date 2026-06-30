import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os

@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_path, 'model_sentimen.pkl')
    vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

model, vectorizer = load_model()

label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

st.title("🌱 Dashboard Analisis Sentimen Ulasan")
st.write("Menggunakan Model Terbaik: **Naive Bayes + N-Gram**")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (Upload CSV)"])

with tab1:
    st.subheader("Uji Coba Satu Ulasan")
    user_input = st.text_area("Masukkan teks ulasan pelanggan di sini:")
    
    if st.button("Prediksi Sentimen"):
        if user_input.strip() != "":
            vec_text = vectorizer.transform([user_input])
            pred_num = model.predict(vec_text)[0]
            
            hasil = label_map.get(pred_num, pred_num) 
            
            if hasil == 'Positif':
                st.success(f"Hasil Prediksi: **{hasil}** 😃")
            elif hasil == 'Negatif':
                st.error(f"Hasil Prediksi: **{hasil}** 😡")
            else:
                st.info(f"Hasil Prediksi: **{hasil}** 😐")
        else:
            st.warning("Teks ulasan tidak boleh kosong.")

with tab2:
    st.subheader("Uji Coba Banyak Data (Upload CSV)")
    uploaded_file = st.file_uploader("Upload file dataset dalam format CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data yang Di-upload:")
        st.dataframe(df.head(3))
        
        text_col = st.selectbox("Pilih kolom yang berisi Teks Ulasan:", df.columns)
        
        label_col = st.selectbox("Pilih kolom Label Aktual (Pilih 'Tidak Ada' jika hanya ingin memprediksi data baru):", ["Tidak Ada"] + list(df.columns))
        
        if st.button("Proses Prediksi Batch"):
            X_vec = vectorizer.transform(df[text_col].fillna("").astype(str))
            predictions = model.predict(X_vec)
            
            df['Hasil Prediksi Model'] = [label_map.get(p, p) for p in predictions]
            
            st.write("✅ **Hasil Prediksi:**")
            st.dataframe(df[[text_col, 'Hasil Prediksi Model']])
            
            if label_col != "Tidak Ada":
                st.markdown("---")
                st.subheader("📊 Evaluasi Model pada Data Ini")
                
                acc = accuracy_score(df[label_col], predictions)
                st.metric(label="Total Accuracy", value=f"{acc:.4f}")
                
                st.text("Classification Report:")
                st.code(classification_report(df[label_col], predictions, zero_division=0))
