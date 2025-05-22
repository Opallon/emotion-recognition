# Настройки путей
JSON_FILENAME = "expected.json"
MODEL_PATH_YOLO = "yolov11n-face.pt"
MODEL_PATH_RESNET = "3_clas_resnet18_trained.pth"

# Параметры моделей
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (224, 224)
MAX_PREDICTIONS_HISTORY = 10
REPEAT_THRESHOLD = 5

# Классы и маппинги
CLASS_MAPPING = {
    "Mode": "100",
    "Smile": "0",
    "Neutral": "1",
    "Surprise": "2",
    "Embarrassment": "3",
    "CurrentGesture": "221",
}

LABELS = ["Smile", "Neutral", "Surprise", "Embarrassment", "CurrentGesture"]
