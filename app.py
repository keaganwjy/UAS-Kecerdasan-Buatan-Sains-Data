import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from google_play_scraper import reviews, Sort
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="App Sentiment Analyzer", layout="wide")

# --- Fungsi Helper ---
@st.cache_resource
def load_assets():
    # Load model dan tools yang sudah dilatih sebelumnya
    model = load_model('sentiment_model.h5')
    tokenizer = joblib.load('tokenizer.joblib')
    le = joblib.load('label_encoder.joblib')
    return model, tokenizer, le

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_app_id(url):
    # Mengambil ID dari URL (id=com.xxx.xxx)
    try:
        match = re.search(r'id=([a-zA-Z0-9._]+)', url)
        if match:
            return match.group(1)
        return None
    except:
        return None

def get_business_insight(sentiment_counts):
    total = sentiment_counts.sum()
    neg = sentiment_counts.get('Negatif', 0)
    pos = sentiment_counts.get('Positif', 0)
    neu = sentiment_counts.get('Netral', 0)
    
    neg_pct = (neg / total) * 100 if total > 0 else 0
    pos_pct = (pos / total) * 100 if total > 0 else 0
    
    insight = ""
    recommendation = ""
    
    if pos_pct > 70:
        insight = "🟢 **Kondisi Sangat Baik:** Mayoritas pengguna menyukai aplikasi ini."
        recommendation = "Fokus pada retensi pengguna dan **upselling** item dalam aplikasi. Pertahankan performa server dan update konten secara berkala agar pengguna tidak bosan."
    elif neg_pct > 40:
        insight = "🔴 **Kondisi Kritis:** Terlalu banyak sentimen negatif."
        recommendation = "Prioritas utama adalah **perbaikan bug/crash** dan optimasi performa. Tim Developer harus segera mengecek log error. Tunda fitur baru, fokus pada stabilitas. Tim CS harus lebih responsif membalas keluhan."
    elif neu > 40:
        insight = "🟡 **Kondisi Stagnan:** Banyak pengguna merasa biasa saja (Netral)."
        recommendation = "Aplikasi berjalan baik tapi kurang 'Wow Factor'. Coba berikan **insentif/hadiah login** atau perbaiki UI/UX agar lebih menarik. Lakukan survei fitur apa yang paling diinginkan."
    else:
        insight = "🔵 **Kondisi Campuran:** Opini pengguna terbelah."
        recommendation = "Analisa lebih dalam komentar negatif spesifik. Biasanya ini masalah kompatibilitas device tertentu atau fitur 'pay-to-win' yang terlalu agresif."
        
    return insight, recommendation

# --- UI Aplikasi ---
st.title("📱 Google Play Store Sentiment Analysis")
st.markdown("Analisis sentimen komentar aplikasi menggunakan **Deep Learning** untuk keputusan bisnis yang lebih baik.")

# Sidebar Input
with st.sidebar:
    st.header("Input Data")
    url_input = st.text_input("Masukkan Link Google Play Store", placeholder="https://play.google.com/store/apps/details?id=...")
    analyze_btn = st.button("Analisa Sentimen")
    
    st.info("Pastikan link memiliki format `id=com.nama.paket`")

# Logika Utama
if analyze_btn and url_input:
    app_id = extract_app_id(url_input)
    
    if not app_id:
        st.error("Link tidak valid atau App ID tidak ditemukan.")
    else:
        with st.spinner(f"Sedang mengambil data untuk ID: {app_id}..."):
            try:
                # 1. Scraping Data Baru
                result, _ = reviews(
                    app_id,
                    lang='id', # Pastikan konsisten dengan model training
                    country='id',
                    sort=Sort.NEWEST,
                    count=100000 
                )
                
                if len(result) == 0:
                    st.warning("Tidak ada ulasan ditemukan untuk aplikasi ini.")
                else:
                    new_df = pd.DataFrame(result)
                    st.success(f"Berhasil mengambil {len(new_df)} ulasan terbaru!")

                    # 2. Load Model & Preprocessing
                    model, tokenizer, le = load_assets()
                    
                    # Bersihkan text
                    new_df['cleaned_content'] = new_df['content'].apply(clean_text)
                    
                    # Tokenize & Padding
                    seq = tokenizer.texts_to_sequences(new_df['cleaned_content'])
                    padded = pad_sequences(seq, maxlen=100, truncating='post')
                    
                    # 3. Prediksi
                    predictions = model.predict(padded)
                    predicted_indices = np.argmax(predictions, axis=1)
                    predicted_labels = le.inverse_transform(predicted_indices)
                    
                    new_df['sentiment_pred'] = predicted_labels
                    
                    # --- Hasil Analisis ---
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Visualisasi Sentimen")
                        
                        # Hitung distribusi
                        sentiment_counts = new_df['sentiment_pred'].value_counts()
                        
                        # Pie Chart
                        fig1, ax1 = plt.subplots()
                        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
                        ax1.axis('equal')  
                        st.pyplot(fig1)
                        
                        # Bar Chart
                        st.bar_chart(sentiment_counts)

                    with col2:
                        st.subheader("💡 Keputusan Bisnis")
                        insight, recommendation = get_business_insight(sentiment_counts)
                        
                        st.markdown(f"**Analisis:**\n{insight}")
                        st.divider()
                        st.markdown(f"**Rekomendasi Tindakan:**\n{recommendation}")
                        

                    # Tampilkan Data Mentah (Opsional)
                    with st.expander("Lihat Data Mentah Hasil Prediksi"):
                        st.dataframe(new_df[['userName', 'content', 'score', 'sentiment_pred']])

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.warning("Pastikan Anda terhubung ke internet dan ID aplikasi benar.")

elif analyze_btn and not url_input:
    st.warning("Silakan masukkan URL terlebih dahulu.")