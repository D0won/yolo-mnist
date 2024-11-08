from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Yolo11 모델 로드
model = YOLO("model/yolo11n-cls.pt")

@app.get("/")
async def index(request: Request) :
    return templates.TemplateResponse("index.html", {"request" : request})

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLO 모델을 사용하여 예측 수행
            results = model(frame)
            for result in results:
                # 예측 결과를 프레임에 표시
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{box.cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")