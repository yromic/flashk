import os
import time # Impor untuk anti-cache
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Konfigurasi Awal ---
app = Flask(__name__)
model_path = 'plant_cnn_final.h5'
upload_folder = 'static/uploads'

# Buat folder upload jika belum ada
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

# --- [BAGIAN A]: PEMUATAN MODEL ---
# Pastikan model.h5 ada di folder yang sama dengan app.py
try:
    model = load_model(model_path)
    print(f"Model '{model_path}' berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model '{model_path}'.")
    print(f"Pastikan file ada dan kamu sudah install tensorflow.")
    print(f"Detail Error: {e}")
    # Jika model gagal dimuat, kita tidak bisa lanjut
    model = None 

# --- [BAGIAN B]: URUTAN NAMA KELAS (SANGAT PENTING!) ---
# Urutan ini HARUS SAMA PERSIS dengan output di Cell 5 Colab kamu.
# Buka Colab, jalankan Cell 5, dan lihat output `print(class_names)`.
#
# Jika output Colab: ['Sakit', 'Sehat'] -> Gunakan ['Sakit', 'Sehat']
# Jika output Colab: ['Sehat', 'Sakit'] -> Gunakan ['Sehat', 'Sakit']
#
# (Saya tebak urutannya 'Sakit' duluan karena alfabetis 'a' vs 'e')
class_names = ['Sehat', 'Sakit']
print(f"Nama kelas yang digunakan: {class_names}")

# --- Rute Halaman Utama ---
@app.route('/')
def index():
    """Menampilkan halaman utama (index.html)."""
    return render_template('index.html')

# --- Rute Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    """Menangani upload file dan melakukan prediksi."""
    
    if model is None:
        # Jika model gagal dimuat saat startup
        return render_template('index.html', error="Error: Model tidak dapat dimuat. Cek log server.")

    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file yang diunggah")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="Nama file kosong, silakan pilih file")
    
    if file:
        try:
            # 1. Simpan file
            # Tambahkan timestamp untuk menghindari masalah cache di browser
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 2. Proses Gambar
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            
            # --- [BAGIAN C]: PERBAIKAN KRITIS (PREPROCESSING) ---
            # JANGAN tambahkan `/ 255.0` di sini.
            # Model .h5 kita SUDAH memiliki layer Rescaling di dalamnya.
            # Kita harus memberinya input piksel mentah [0, 255].
            img_batch = np.expand_dims(img_array, axis=0) 

            # 3. Lakukan Prediksi
            predictions = model.predict(img_batch)
            
            # 4. Dapatkan Hasil
            score = np.max(predictions[0]) * 100
            pred_index = np.argmax(predictions[0])
            pred_class = class_names[pred_index] # Ambil nama dari list

            # 5. Tampilkan hasil
            return render_template('result.html', 
                                   filename=filename, # Kirim nama file baru (dengan timestamp)
                                   label=pred_class, 
                                   confidence=f"{score:.2f}")
        
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            return render_template('index.html', error=f"Terjadi error: {e}")
    
    return render_template('index.html', error="Terjadi error saat upload")

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)

