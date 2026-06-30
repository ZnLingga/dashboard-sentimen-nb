import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 0.5px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
    }
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 6px solid;
    }
    .card-pos { background-color: #E8F5E9; border-left-color: #2E7D32; color: #1B5E20; }
    .card-neu { background-color: #FFF8E1; border-left-color: #F57F17; color: #E65100; }
    .card-neg { background-color: #FFEBEE; border-left-color: #C62828; color: #B71C1C; }
    
    .card-title { font-size: 0.95rem; font-weight: 500; margin: 0; opacity: 0.8; }
    .card-value { font-size: 2rem; font-weight: 700; margin: 5px 0 0 0; }
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
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=90)
    st.markdown("### ⚙️ Sistem Cerdas")
    st.markdown("---")
    st.markdown("💡 **Arsitektur Pemodelan:**")
    st.markdown("- **Algoritma:** Naive Bayes Classifier")
    st.markdown("- **Ekstraksi Fitur:** N-Gram Text Representation")
    st.markdown("- **Akurasi Baseline:** `85.81%`")
    st.markdown("---")
    st.caption("Dashboard ini dirancang untuk mendeteksi emosi dan kecenderungan ulasan konsumen secara otomatis.")

st.markdown('<p class="main-title">✨ DASHBOARD ANALISIS SENTIMEN ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Komputasi Analisis Opini Konsumen Berbasis Machine Learning</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Single Prediction", "📊 Batch Prediction (Analisis Data Massal)"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📝 Masukkan Kalimat Ulasan")
        user_input = st.text_area("Ketik atau tempel ulasan konsumen di bawah ini:", height=160, placeholder="Tulis komentar di sini...")
        submit_btn = st.button("🔮 Mulai Ekstraksi Sentimen", use_container_width=True)
        
    with col2:
        st.markdown("### 🎯 Hasil Klasifikasi")
        if submit_btn:
            if user_input.strip() != "":
                vec_text = vectorizer.transform([user_input])
                pred_num = model.predict(vec_text)[0]
                hasil = label_map.get(pred_num, pred_num)
                
                if hasil == 'Positif':
                    st.success(f"### Sentimen: {hasil} ✨\n\nKomentar menunjukkan tingkat kepuasan yang tinggi.")
                elif hasil == 'Negatif':
                    st.error(f"### Sentimen: {hasil} 🚨\n\nKomentar terdeteksi berisi keluhan atau ketidakpuasan.")
                else:
                    st.warning(f"### Sentimen: {hasil} ⚖️\n\nKomentar bersifat netral atau informasi umum.")
            else:
                st.warning("⚠️ Kolom teks ulasan tidak boleh kosong.")
        else:
            st.info("Sistem siap. Silakan masukkan kalimat ulasan di kolom sebelah kiri.")

with tab2:
    st.markdown("### 📂 Unggah Berkas Dataset")
    uploaded_file = st.file_uploader("Silakan unggah dokumen ulasan konsumen dalam format (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            text_col = st.selectbox("🎯 Pilih Kolom Teks Ulasan:", df.columns)
        with c_sel2:
            label_col = st.selectbox("🔑 Pilih Kolom Target Aktual (Opsional):", ["Tidak Ada"] + list(df.columns))
            
        if st.button("⚙️ Jalankan Komputasi Massal", type="primary", use_container_width=True):
            with st.spinner('Sedang menganalisis dataset...'):
                X_vec = vectorizer.transform(df[text_col].fillna("").astype(str))
                predictions = model.predict(X_vec)
                df['Hasil Prediksi Model'] = [label_map.get(p, p) for p in predictions]
                
                counts = df['Hasil Prediksi Model'].value_counts()
                total_pos = counts.get('Positif', 0)
                total_neu = counts.get('Netral', 0)
                total_neg = counts.get('Negatif', 0)
            
            st.markdown("---")
            st.markdown("### 📈 Ringkasan & Distribusi Hasil Analisis")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f'<div class="metric-card card-pos"><p class="card-title">🟢 TOTAL SENTIMEN POSITIF</p><p class="card-value">{total_pos} Ulasan</p></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="metric-card card-neu"><p class="card-title">🟡 TOTAL SENTIMEN NETRAL</p><p class="card-value">{total_neu} Ulasan</p></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-card card-neg"><p class="card-title">🔴 TOTAL SENTIMEN NEGATIF</p><p class="card-value">{total_neg} Ulasan</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            g_col1, g_col2 = st.columns([1, 1])
            
            with g_col1:
                st.markdown("#### 📊 Grafik Komparasi Frekuensi")
                st.bar_chart(counts, color="#1E3A8A")
                
            with g_col2:
                if label_col != "Tidak Ada":
                    st.markdown("#### 🏆 Metrik Evaluasi Validasi")
                    
                    y_true_safe = df[label_col].astype(str)
                    y_pred_safe = pd.Series(predictions).astype(str)
                    acc = accuracy_score(y_true_safe, y_pred_safe)
                    
                    col_inner1, col_inner2 = st.columns(2)
                    col_inner1.metric(label="🎯 Skor Akurasi Valid", value=f"{acc*100:.2f}%")
                    col_inner2.metric(label="📦 Total Sampel Sukses", value=f"{len(df)} Data")
                    
                    st.markdown("#### 🧩 Confusion Matrix")
                    
                    label_urut = ['Negatif', 'Netral', 'Positif']
                    cm = confusion_matrix(y_true_safe, y_pred_safe, labels=label_urut)
                    
                    cm_df = pd.DataFrame(cm, index=label_urut, columns=label_urut).reset_index().melt(id_vars='index')
                    cm_df.columns = ['Aktual', 'Prediksi', 'Jumlah']
                    
                    base = alt.Chart(cm_df).encode(
                        x=alt.X('Prediksi:O', title='Prediksi Model'),
                        y=alt.Y('Aktual:O', title='Label Aktual', sort='-y')
                    )
                    rect = base.mark_rect().encode(
                        color=alt.Color('Jumlah:Q', scale=alt.Scale(scheme='blues'), title='Jumlah Data')
                    )
                    text = base.mark_text(baseline='middle', fontSize=14, fontWeight='bold').encode(
                        text='Jumlah:Q',
                        color=alt.condition(
                            alt.datum.Jumlah > (cm_df['Jumlah'].max() / 2),
                            alt.value('white'),
                            alt.value('black')
                        )
                    )
                    cm_chart = (rect + text).properties(height=300)
                    
                    st.altair_chart(cm_chart, use_container_width=True)
                    
                    with st.expander("Buka Detail Laporan Klasifikasi (Precision/Recall)"):
                        st.code(classification_report(y_true_safe, y_pred_safe, zero_division=0))
                else:
                    st.markdown("#### 💡 Informasi Kluster")
                    st.info("Unggah berkas yang memiliki kunci label aktual jika Anda ingin melihat persentase kalkulasi akurasi model serta Confusion Matrix secara otomatis pada dataset ini.")

            st.markdown("---")
            
            st.markdown("#### 📋 Lembar Pratinjau Dataset Hasil Prediksi")
            def style_sentiment_column(val):
                if val == 'Positif': return 'background-color: #D1E7DD; color: #0F5132; font-weight: 500;'
                elif val == 'Negatif': return 'background-color: #F8D7DA; color: #842029; font-weight: 500;'
                elif val == 'Netral': return 'background-color: #FFF3CD; color: #664D03; font-weight: 500;'
                return ''
            
            styled_df = df[[text_col, 'Hasil Prediksi Model']].head(50).style.map(style_sentiment_column, subset=['Hasil Prediksi Model'])
            st.dataframe(styled_df, use_container_width=True, height=400)
