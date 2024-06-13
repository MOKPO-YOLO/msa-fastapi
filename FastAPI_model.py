from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# YOLO 모델 로드
model = YOLO(r'C:\Users\User\yoloray\msa-fastapi\models\v1.0.1_240613.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read() # 파일 읽어오기
    image = Image.open(io.BytesIO(contents)).convert("RGB") # RGB 형식 변환
    
    # numpy 형태로 업로드
    image_np = np.array(image)
    
    # detection 진행
    results = model(image_np)
    
    # json 형태 변환
    predictions = []
    for result in results:
        for *box, conf, cls in result:
            predictions.append({
                "box": box,
                "confidence": conf,
                "class": int(cls)
            })
    
    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # 메인
