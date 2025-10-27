import os
import time # Impor untuk anti-cache
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
model_path = 'plant_cnn_final.h5'
upload_folder = 'static/uploads'


os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder

try:
    model = load_model(model_path)
    print(f"Model '{model_path}' berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model '{model_path}'.")
    print(f"Pastikan file ada dan kamu sudah install tensorflow.")
    print(f"Detail Error: {e}")

    model = None 

class_names = ['Sehat', 'Sakit']
print(f"Nama kelas yang digunakan: {class_names}")

@app.route('/')
def index():
    """Menampilkan halaman utama (index.html)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Menangani upload file dan melakukan prediksi."""
    
    if model is None:

        return render_template('index.html', error="Error: Model tidak dapat dimuat. Cek log server.")

    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file yang diunggah")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="Nama file kosong, silakan pilih file")
    
    if file:
        try:

            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
       
            img_batch = np.expand_dims(img_array, axis=0) 

            predictions = model.predict(img_batch)

            score = np.max(predictions[0]) * 100
            pred_index = np.argmax(predictions[0])
            pred_class = class_names[pred_index] # Ambil nama dari list


            return render_template('result.html', 
                                   filename=filename, # Kirim nama file baru (dengan timestamp)
                                   label=pred_class, 
                                   confidence=f"{score:.2f}")
        
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            return render_template('index.html', error=f"Terjadi error: {e}")
    
    return render_template('index.html', error="Terjadi error saat upload")

if __name__ == '__main__':
    app.run(debug=True)

