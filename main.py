'''
Petunjuk penggunaan aplikasi:
- Menggunakan Python 3.11.3

- Untuk menginstal library yang dibutuhkan, jalankan di terminal:
pip install -r requirements.txt

- Untuk menjalankan aplikasi, jalankan di terminal:
uvicron main:app
'''



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# Konfigurasi aplikasi
known_faces_folder = "known_faces"

if not os.path.exists(os.path.join(os.getcwd(), known_faces_folder)):
    raise HTTPException(status_code=400, detail=f"Folder {known_faces_folder} untuk menyimpan kumpulan wajah tidak ditemukan.")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model diunduh dari internet secara otomatis dan disimpan ke dalam folder 'models' yang juga dibuatkan secara otomatis
ctx_id = -1
face_analyzer = FaceAnalysis(name='buffalo_s', root='.', providers=['CUDAExecutionProvider' if ctx_id == 0 else 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=ctx_id) # 0 = GPU slot pertama, -1 = CPU




# Menyimpan embedding muka (hanya file .jpg) dari known_faces_folder
# Jika ada nama file yang sama, maka file yang selanjutnya akan menimpa yang awal
known_faces = {}

def load_known_faces():
    known_files = [f for f in os.listdir(known_faces_folder) if os.path.isfile(os.path.join(known_faces_folder, f))]

    for file in known_files:
        if file.lower().endswith((".jpg")):
            username = os.path.splitext(file)[0]
            image = cv2.imread(os.path.join(known_faces_folder, file))
            faces = face_analyzer.get(image)

            if faces:
                known_faces[username] = faces[0].embedding
                print(f"[INFO] Berhasil memuat face embedding untuk: {username}")

load_known_faces() # Dijalankan ketika pertama kali menjalankan aplikasi FastAPI



# Proses menebak wajah yang diinput
'''
Petunjuk penggunaan Postman:
1. Method: POST

2. URL: http://127.0.0.1:8000/api/recognize

3. Body -> form-data:
    - KEY = Type: File -> field_name: file; VALUE = Masukkan salah satu foto dari folder 'Semua Foto'

4. Send
'''
@app.post("/api/recognize")
async def recognize_face(file: UploadFile = File(...)):
    # Cek ekstensi file
    file_extension = os.path.splitext(file.filename)[-1].lower()

    if file_extension != '.jpg':
        raise HTTPException(status_code=400, detail="Hanya menerima ekstensi file .jpg")

    # Cek isi file yang diunggah
    input_data = await file.read()

    if not input_data:
        raise HTTPException(status_code=400, detail="File yang diunggah tidak berisi.")

    np_data = np.frombuffer(input_data, np.uint8)
    input_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    input_faces = face_analyzer.get(input_img) # bbox, kps, det_score, landmark_3d_68, pose, landmark_2d_106, gender, age, embedding

    if not input_faces:
        raise HTTPException(status_code=400, detail="Tidak ada wajah yang terdeteksi (liveness model).")
    
    # Mencocokkan muka yang ingin dikenali dengan kumpulan wajah yang telah dikenali
    input_embedding = input_faces[0].embedding.reshape(1, -1)
    best_match_username = None
    best_score = -1

    for username, known_embedding in known_faces.items():
        known_embedding = known_embedding.reshape(1, -1)
        score = cosine_similarity(input_embedding, known_embedding)[0][0]

        if score > best_score:
            best_score = score
            best_match_username = username

    if best_score > 0.6:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return JSONResponse({"status": "success", "username": best_match_username, "time": timestamp})
    else:
        raise HTTPException(status_code=404, detail="Wajah tidak dikenali.")



# Menyimpan gambar yang diunggah ke dalam known_faces_folder dengan format username.jpg
# Diharapkan file yang diunggah dalam format file untuk foto (.jpg atau .png) -> Tangani exception-nya dari sisi web
# Di database, harap gunakan constraint unique pada field username agar tidak menimpa file foto dan face embedding yang sudah tersimpan
'''
Petunjuk penggunaan Postman:
1. Method: POST

2. URL: http://127.0.0.1:8000/api/upload_face

3. Body -> form-data:
    - KEY = Type: File -> field_name: file; VALUE = Masukkan salah satu foto dari folder 'Semua Foto' yang nama foto tersebut berupa username untuk disimpan

4. Send
'''
@app.post("/api/upload_face")
async def upload_face(file: UploadFile = File(...)):
    # Cek ekstensi file
    file_extension = os.path.splitext(file.filename)[-1].lower()

    if file_extension != '.jpg':
        raise HTTPException(status_code=400, detail="Hanya menerima ekstensi file .jpg")

    # Cek isi file yang diunggah
    new_file = await file.read()

    if not new_file:
        raise HTTPException(status_code=400, detail="File yang diunggah tidak berisi.")

    # Simpan foto baru dengan nama file berupa username dan formatnya .jpg
    new_username = file.filename.split('.')[0]
    filename = f"{new_username}{file_extension}"
    file_path = os.path.join(known_faces_folder, filename)

    with open(file_path, "wb") as f:
        f.write(new_file)

    # Mengekstrak face embedding untuk foto baru dan menambahkannya ke dalam daftar wajah yang sudah dikenali (objek known_faces)
    new_image = cv2.imdecode(np.frombuffer(new_file, np.uint8), cv2.IMREAD_COLOR)
    faces = face_analyzer.get(new_image)

    if faces:
        known_faces[new_username] = faces[0].embedding
    else:
        raise HTTPException(status_code=400, detail="Tidak ada muka yang terdeteksi.")

    return JSONResponse({"status": "success", "message": f"Wajah {new_username} berhasil disimpan dan dikenali."})