# Python 3.11.3
# pip install -r requirements.txt
# uvicron main:app
# Model diunduh dari internet secara otomatis dan disimpan ke dalam folder 'models' yang juga dibuatkan secara otomatis
# Postman:
#   - Method: POST
#   - URL: http://127.0.0.1:8000/api/recognize
#   - Body -> form-data -> KEY = file -> VALUE = masukkan salah satu foto dari folder 'Semua Foto' -> Send


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
import cv2
import os
# import torch
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis



app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ctx_id = 0 if torch.cuda.is_available() else -1
ctx_id = -1
face_analyzer = FaceAnalysis(name='buffalo_s', root='.', providers=['CUDAExecutionProvider' if ctx_id == 0 else 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=ctx_id) # 0 = GPU slot pertama, -1 = CPU




known_faces = {} # "Saiful": [ 1.321312..., 2.12314..., ... ]
# attendance_log = []

def load_known_faces():
    for file in os.listdir("known_faces"):
        if file.lower().endswith((".jpg", ".png")):
            name = os.path.splitext(file)[0]
            image = cv2.imread(os.path.join("known_faces", file))
            faces = face_analyzer.get(image)

            if faces:
                known_faces[name] = faces[0].embedding
                print(f"[INFO] Loaded face embedding for: {name}")

load_known_faces()



@app.post("/api/recognize")
async def recognize_face(file: UploadFile = File(...)):
    input_data = await file.read()
    np_data = np.frombuffer(input_data, np.uint8)
    input_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    input_faces = face_analyzer.get(input_img) # # bbox, kps, det_score, landmark_3d_68, pose, landmark_2d_106, gender, age, embedding
    print(f"\n======= Input image\n{input_faces}\n=======\n")

    if not input_faces:
        raise HTTPException(status_code=400, detail="No face detected (liveness model).")
    
    in_embedding = input_faces[0].embedding.reshape(1, -1)
    best_match_name = None
    best_score = -1

    for name, known_embedding in known_faces.items():
        known_embedding = known_embedding.reshape(1, -1)
        score = cosine_similarity(in_embedding, known_embedding)[0][0]

        if score > best_score:
            best_score = score
            best_match_name = name

    if best_score > 0.6:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # attendance_log.append({"name": best_match_name, "time": timestamp})

        return JSONResponse({"status": "success", "name": best_match_name, "time": timestamp})
    else:
        raise HTTPException(status_code=404, detail="Face not recognized.")



# @app.get("/api/attendance")
# def get_attendance():
#     return attendance_log