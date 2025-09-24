from inference import handDetector, process_face, process_hand
import cv2
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import mediapipe as mp
import json

from video_capture import VideoCaptureClass


def main_loop():
    """
    Основной цикл:
      - Открываем камеру
      - В бесконечном цикле получаем кадр
      - Считываем data.json, определяем режим (1 или 2)
      - В зависимости от режима обрабатываем кадр
      - Показываем результат в окне
      - По нажатию ESC выходим
    """

    # Инициализация YOLO
    model_path_YOLO = "yolov11n-face.pt"
    model_YOLO = YOLO(model_path_YOLO)

    # Инициализируем ResNet18
    model_resnet = models.resnet18(pretrained=True)
    model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, 3)
    try:
        model_resnet.load_state_dict(torch.load("3_clas_resnet18_trained.pth"))
    except Exception as e:
        print(f"Ошибка загрузки весов ResNet: {e}")
    model_resnet.eval()

    # Трансформации для ResNet
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Инициализация MediaPipe (для определения руки в process_face)
    hands_for_face = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Детектор рук (используется в режиме data=2)
    hand_detector = handDetector()
    video_capture = VideoCaptureClass()
    while True:
        success, frame = video_capture.read_capture()
        if not success:
            print("Ошибка при получении кадра")
            break

        # Считываем JSON (который определяет режим)
        with open("expected.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        if data["Mode"] == 1:
            # YOLO + ResNet + проверка руки
            process_face(
                frame, model_YOLO, model_resnet, transform, hands_for_face, data["Mode"]
            )
        elif data["Mode"] == 2:
            # Режим распознавания руки c hand_detector
            process_hand(frame, hand_detector, data["Mode"])
        else:
            print("Ошибка ввода в expected.json")

        cv2.imshow("Camera", frame)

        # Выходим по нажатию ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Закрываем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    hands_for_face.close()


if __name__ == "__main__":
    main_loop()
