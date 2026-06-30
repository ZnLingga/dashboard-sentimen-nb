import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

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

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100) # Ikon ilustrasi
    st.title("🤖 Info Model")
    st.info("**Algoritma:** Naive Bayes\n\n**Fitur Teks:** N-Gram\n\n**Akurasi Tertinggi:** 85.81%")
    st.markdown("---")
    st.write("📌 **Cara Penggunaan:**")
    st.write("1. Gunakan **Single Prediction** untuk mencoba satu kalimat.")
    st.write("2. Gunakan **Batch Prediction** untuk menguji file CSV berisi banyak ulasan sekaligus.")

st.markdown('<p class="main-title">✨ Dashboard Analisis Sentimen ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Evaluasi Cerdas Ulasan Pelanggan Menggunakan Machine Learning</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Single Prediction", "📂 Batch Prediction (Upload CSV)"])

with tab1:
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.markdown("### 📝 Masukkan Teks Ulasan")
        user_input = st.text_area("Ketik atau paste ulasan pelanggan di bawah ini:", height=150, placeholder="Contoh: Produknya luar biasa, pengirimannya juga sangat cepat!")
        submit_btn = st.button("🚀 Prediksi Sentimen", use_container_width=True)
        
    with col2:
        st.markdown("### 🎯 Hasil Prediksi")
        if submit_btn:
            if user_input.strip() != "":
                vec_text = vectorizer.transform([user_input])
                pred_num = model.predict(vec_text)[0]
                hasil = label_map.get(pred_num, pred_num)
                
                if hasil == 'Positif':
                    st.success(f"### {hasil} 😃\nModel mendeteksi sentimen positif.")
                elif hasil == 'Negatif':
                    st.error(f"### {hasil} 😡\nModel mendeteksi sentimen negatif.")
                else:
                    st.warning(f"### {hasil} 😐\nModel mendeteksi sentimen netral.")
            else:
                st.warning("⚠️ Teks ulasan tidak boleh kosong.")
        else:
            st.info("Silakan masukkan teks dan tekan tombol prediksi di samping.")

with tab2:
    st.markdown("### 📊 Prediksi Sekaligus Banyak Data")
    uploaded_file = st.file_uploader("Upload dataset ulasan Anda (Format: .csv)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        col_text, col_label = st.columns(2)
        with col_text:
            text_col = st.selectbox("Pilih kolom yang berisi **Teks Ulasan**:", df.columns)
        with col_label:
            label_col = st.selectbox("Pilih kolom **Label Aktual** (Jika hanya ingin memprediksi, pilih 'Tidak Ada'):", ["Tidak Ada"] + list(df.columns))
            
        if st.button("⚙️ Proses Prediksi Batch", type="primary"):
            with st.spinner('Memproses data...'):
                X_vec = vectorizer.transform(df[text_col].fillna("").astype(str))
                predictions = model.predict(X_vec)
                df['Hasil Prediksi'] = [label_map.get(p, p) for p in predictions]
                
            st.markdown("---")
            
            st.markdown("### 📈 Visualisasi Sentimen")
            sentiment_counts = df['Hasil Prediksi'].value_counts()
            st.bar_chart(sentiment_counts, color="#3498DB")
            
            st.markdown("### 📋 Preview Hasil Data")
            st.dataframe(df[[text_col, 'Hasil Prediksi']].head(15), use_container_width=True)
            
            if label_col != "Tidak Ada":
                st.markdown("---")
                st.markdown("### 🏆 Evaluasi Model")
                
                y_true_safe = df[label_col].astype(str)
                y_pred_safe = pd.Series(predictions).astype(str)
                
                acc = accuracy_score(y_true_safe, y_pred_safe)
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(label="✅ Akurasi Total", value=f"{acc*100:.2f}%")
                col_m2.metric(label="📦 Total Data Diuji", value=f"{len(df)} Ulasan")
                
                with st.expander("Lihat Classification Report Lengkap"):
                    st.code(classification_report(y_true_safe, y_pred_safe, zero_division=0))
