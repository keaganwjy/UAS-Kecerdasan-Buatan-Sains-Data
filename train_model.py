import pandas as pd
import numpy as np
import re
from google_play_scraper import reviews, Sort
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

# --- 1. Konfigurasi & Scraping Data ---
APP_ID = 'com.bladetw.bd'
print(f"Sedang mengambil data mentah untuk App ID: {APP_ID}...")

# Ambil data banyak agar LSTM bisa belajar pola kalimat
result, _ = reviews(
    APP_ID,
    lang='id', 
    country='id',
    sort=Sort.NEWEST, 
    count=36800
)

df = pd.DataFrame(result)
print(f"Total data mentah: {len(df)}")

# --- 2. Labeling (Sesuai Request) ---
def categorize_sentiment(score):
    if score <= 2:
        return 'Negatif'   # Score 1-2
    elif score == 3:
        return 'Netral'    # Score 3
    else:
        return 'Positif'   # Score 4-5

df['sentiment'] = df['score'].apply(categorize_sentiment)

# --- 3. Balancing Data (PENTING untuk akurasi Negatif) ---
df_neg = df[df['sentiment'] == 'Negatif']
df_neu = df[df['sentiment'] == 'Netral']
df_pos = df[df['sentiment'] == 'Positif']

min_count = min(len(df_neg), len(df_neu), len(df_pos))
print(f"Menyeimbangkan data menjadi {min_count} sampel per kategori...")

df_neg_balanced = df_neg.sample(min_count, random_state=42)
df_neu_balanced = df_neu.sample(min_count, random_state=42)
df_pos_balanced = df_pos.sample(min_count, random_state=42)

df_balanced = pd.concat([df_neg_balanced, df_neu_balanced, df_pos_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 4. Advanced NLP Preprocessing ---

# Daftar Stopwords Indonesia sederhana (Kata umum yang tidak punya sentimen)
stopwords_id = set([
    "dan", "yang", "di", "ke", "dari", "ini", "itu", "untuk", "pada", 
    "dengan", "adalah", "ya", "orang", "karena", "juga", "jadi", "lagi"
])

def clean_text_nlp(text):
    text = str(text).lower()
    # Hapus karakter selain huruf dan angka
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenisasi manual sederhana (split spasi)
    words = text.split()
    # Hapus stopwords (Teknik NLP agar model fokus ke kata sifat)
    words = [w for w in words if w not in stopwords_id]
    # Gabung lagi
    return " ".join(words)

df_balanced['cleaned_content'] = df_balanced['content'].apply(clean_text_nlp)

X = df_balanced['cleaned_content'].values
y = df_balanced['sentiment'].values

# Encode Label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Pastikan urutan: Negatif=0, Netral=1, Positif=2 (biasanya urut abjad)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- 5. Tokenization & Padding ---
vocab_size = 5000
embedding_dim = 32 # Dinaikkan sedikit untuk menampung nuansa makna
max_length = 100
oov_tok = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

training_sequences = tokenizer.texts_to_sequences(X_train)
testing_sequences = tokenizer.texts_to_sequences(X_test)

training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length, truncating='post', padding='post')
testing_padded = tf.keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_length, truncating='post', padding='post')

# --- 6. Model Deep Learning dengan LSTM (The Upgrade) ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # Layer Bidirectional LSTM
    # Bidirectional artinya dia membaca kalimat dari depan ke belakang DAN belakang ke depan
    # Ini sangat ampuh untuk memahami konteks kalimat
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("\nSedang melatih model LSTM (akan sedikit lebih lama tapi lebih pintar)...")
model.fit(training_padded, y_train, epochs=15, validation_data=(testing_padded, y_test), verbose=1)

# --- 7. Simpan Model ---
model.save('sentiment_model.h5')
joblib.dump(tokenizer, 'tokenizer.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\n✅ SELESAI! Model LSTM telah disimpan.")

# --- 8. Evaluasi Model (Classification Report & Confusion Matrix) ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Lakukan Prediksi pada Data Test
print("\nSedang melakukan evaluasi pada data test...")
y_pred_probs = model.predict(testing_padded)
y_pred = np.argmax(y_pred_probs, axis=1) # Mengambil index kelas dengan probabilitas tertinggi

# Ambil nama label asli dari LabelEncoder agar laporan mudah dibaca
# Urutan biasanya alfabetis: ['Negatif', 'Netral', 'Positif']
target_names = [str(cls) for cls in le.classes_]

# 2. Tampilkan Classification Report
print("\n" + "="*30)
print("CLASSIFICATION REPORT")
print("="*30)
print(classification_report(y_test, y_pred, target_names=target_names))

# 3. Tampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title('Confusion Matrix - Analisis Sentimen')
plt.ylabel('Label Asli (Aktual)')
plt.xlabel('Label Prediksi Model')
plt.show()