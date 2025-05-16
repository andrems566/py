import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch

# Verifica se há GPU
gpu_ok = torch.cuda.is_available()
device = 'cuda' if gpu_ok else 'cpu'
print("Usando:", device)

model = YOLO("Yolo/yolov8n.pt")

# Inicializa webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()
#parte da condição ligar o scanner
plot_ativo = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    frame = cv2.flip(frame, 1)

    results = model.predict(source=frame, device=device, verbose=False)

    # Anotações no frame
    annotated_frame = results[0].plot() if plot_ativo else frame

    cv2.imshow("YOLOv8", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        plot_ativo = True
    elif key == ord("0"):
        plot_ativo = False

cap.release()
cv2.destroyAllWindows()
