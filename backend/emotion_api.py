from fastapi import FastAPI, UploadFile, File
from models.speech_emotion_recognition.ser_wrapper import predict_emotion_from_audio
import shutil

app = FastAPI()

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    emotion = predict_emotion_from_audio("temp.wav")
    return {"emotion": emotion}
