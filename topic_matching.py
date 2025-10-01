'''
Petunjuk penggunaan aplikasi:
- Menggunakan Python 3.11.3

- Untuk menginstal library yang dibutuhkan, jalankan di terminal:
pip install -r requirements.txt

- Ekstrak file models.zip

- Untuk menjalankan aplikasi, jalankan di terminal (topic_matching adalah nama file ini tanpa ekstensi .py):
uvicron topic_matching:app

'''


from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json



app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

local_model_dir = os.path.join(".", "models", "multilingual-e5-small")
onnx_model_path = os.path.join(local_model_dir, "onnx", "model.onnx")



# Tokenisasi, inference, masking + mean pooling, dan normalisasi
def get_embeddings(texts, tokenizer, session):
    if isinstance(texts, str):
        texts = [texts]

    # Tokenisasi
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    onnx_inputs = {"input_ids": inputs["input_ids"].astype(np.int64), "attention_mask": inputs["attention_mask"].astype(np.int64)}
    
    if 'token_type_ids' in inputs:
        onnx_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)
    else:
        onnx_inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'], dtype=np.int64)

    # inference
    outputs = session.run(None, onnx_inputs)
    last_hidden_state = outputs[0] # embedding

    # Mask padding tokens dan mean pooling
    input_mask_expanded = np.expand_dims(inputs["attention_mask"], axis=-1)
    masked_hidden = last_hidden_state * input_mask_expanded
    sum_hidden = masked_hidden.sum(axis=1)
    lengths = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    mean_embeddings = sum_hidden / lengths

    # L2 normalization
    norm = np.linalg.norm(mean_embeddings, ord=2, axis=1, keepdims=True)
    normalized_embeddings = mean_embeddings / np.maximum(norm, 1e-10)

    return normalized_embeddings



'''
Petunjuk penggunaan Postman:
1. Method: POST

2. URL: http://127.0.0.1:8000/api/match_topic

3. Body -> form-data:
    - KEY = sentences (Text); VALUE = Sebuah atau sekumpulan string yang tiap string HARUS diapit tanda petik
    - KEY = topic (Text); VALUE = Sebuah string

    - Catatan:
        - Field sentences berisi sebuah atau sekumpulan monolog yang diawali dengan prefiks "query: " dan bagusnya tiap string terdiri dari beberapa kalimat
            - Contoh monolog seseorang:
"query: Tadi saat praktikum, saya mengamati bahwa benda yang dicelupkan ke dalam air mengalami gaya ke atas. Kami menimbang benda sebelum dan sesudah dimasukkan ke dalam air, dan beratnya berkurang karena adanya gaya apung. Ini sesuai dengan Hukum Archimedes yang menyatakan bahwa besar gaya apung sama dengan berat cairan yang dipindahkan oleh benda tersebut."

            - Contoh monolog 4 orang:
[
    "query: Tadi saat praktikum, saya mengamati bahwa benda yang dicelupkan ke dalam air mengalami gaya ke atas. Kami menimbang benda sebelum dan sesudah dimasukkan ke dalam air, dan beratnya berkurang karena adanya gaya apung. Ini sesuai dengan Hukum Archimedes yang menyatakan bahwa besar gaya apung sama dengan berat cairan yang dipindahkan oleh benda tersebut.",
    "query: Kak, saya masih bingung kenapa benda yang dicelupkan ke dalam air jadi terasa lebih ringan ya? Oh, jadi itu sebabnya benda bisa mengapung atau tenggelam ya? Kalau kita ukur berat benda di udara dan di air, selisihnya itu gaya Archimedes ya, Kak?",
    "query: Kak, saya tadi coba celupkan benda ke dalam air, terus saya ukur beratnya di air. Kok beda banget sama waktu ditimbang di udara? Berarti, gaya Archimedes itu yang membuat benda jadi terasa lebih ringan di dalam air, ya? Oh, jadi selisih berat benda di udara dan di dalam air itu adalah nilai gaya Archimedes? Lalu kalau bendanya mengapung, artinya gaya Archimedes lebih besar dari berat benda? Berarti dengan data dari percobaan ini, kita bisa hitung volume benda juga, ya Kak?",
    "query: Kak, sorry banget ya, I still don't get it. Kenapa pas benda gue celupin ke air, it feels like way lighter? So you're saying... air tuh ngedorong balik gitu ya? Okay, makes sense. Jadi waktu gue timbang di air terus angkanya turun, itu karena si buoyant force tadi? Wait, berarti kalo benda bisa ngambang, itu tandanya gaya apungnya lebih besar dari berat bendanya, right? Interesting sih. So technically, dari selisih berat di air sama di udara, kita bisa hitung volume benda? Okay noted, Kak. Ternyata Fisika can be fun juga ya, kalo udah relate ke real life gini."
]

        - Field topic berisi DESKRIPSI topik, yang diawali dengan prefiks "query: ", yang berguna untuk dicocokkan dengan masing-masing monolog dari field sentences. Contoh topic:
            - "query: Fisika: Hukum Archimedes"
            - query: Fisika: Hukum Archimedes

4. Send

5. Output berupa status dan similarities
    - similarities berupa lis nilai tiap string di sentences yang dibandingkan dengan string topic, berurut sesuai urutan string pada sentences
        - similarities[0] = kemiripan string sentences[0] dengan string topic
        - Semakin tingi nilai similarities, semakin dekat hubungan antar dua string tersebut (semakin mirip/cocok)
'''
@app.post("/api/match_topic")
async def match_topic(sentences: str = Form(...), topic: str = Form(...)):
    sentences = json.loads(sentences)

    try:
        # Memuat model
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)

        # Memulai ONNX runtime
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model tidak ditemukan. Pastikan konfigurasi model terletak di dalam direktori yang sesuai dengan isi variabel local_model_dir ({local_model_dir}) dan model ONNX terletak di tempat yang sesuai dengan isi variabel onnx_model_path ({onnx_model_path})")

    sentence_embeddings = get_embeddings(sentences, tokenizer, session)
    topic_embedding = get_embeddings(topic, tokenizer, session)
    similarities = (sentence_embeddings @ topic_embedding.T)[:, 0]
    
    return JSONResponse({"status": "success", "similarities": similarities.tolist()})