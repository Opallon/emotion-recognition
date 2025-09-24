# Emotion Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLO](https://img.shields.io/badge/YOLO-Face%20Detection-green)
![ResNet](https://img.shields.io/badge/ResNet18-Backbone-orange)

**Система для автоматического распознавания эмоций по изображениям лиц**

</div>

## 📋 О проекте

Это современная система компьютерного зрения, которая сочетает детекцию лиц с помощью YOLO и классификацию эмоций с использованием ResNet18. Система анализирует изображения или видео потоки и определяет эмоциональное состояние с высокой точностью.

### 🔍 Рабочий процесс
1. **Детекция лиц**: YOLO обнаруживает и вырезает лица на изображении
2. **Классификация эмоций**: ResNet18 анализирует каждое обнаруженное лицо
3. **Вывод результатов**: Результаты сохраняются в структурированном JSON-формате

## 😊 Распознаваемые эмоции

Система классифицирует 4 основные эмоции:

| Эмоция | Метка | Описание |
|--------|-------|----------|
| **Happy** (Радость) | 1 | Улыбка, смех, положительные эмоции |
| **Neutral** (Нейтральная) | 0 | Спокойное выражение лица |
| **Surprise** (Удивление) | 0 | Широко открытые глаза, удивление |
| **Embarrassment** (Смущение) | 0 | Неловкость, смущение |

## 🚀 Быстрый старт

### Предварительные требования

```bash
# Установите зависимости
pip install torch torchvision opencv-python ultralytics numpy pillow
```

### Использование

```python
from emotion_detector import EmotionDetector

# Инициализация детектора
detector = EmotionDetector()

# Обработка изображения
result = detector.process_image("path/to/image.jpg")

# Результат содержит детектированные лица и эмоции
print(result)
```

### 📄 Пример вывода JSON

```json
{
  "image_path": "path/to/image.jpg",
  "faces_detected": 2,
  "results": [
    {
      "face_id": 0,
      "bbox": [100, 150, 200, 250],
      "emotions": {
        "Happy": 1,
        "Neutral": 0,
        "Surprise": 0,
        "Embarrassment": 0
      },
      "dominant_emotion": "Happy",
      "confidence": 0.95
    },
    {
      "face_id": 1,
      "bbox": [300, 120, 400, 220],
      "emotions": {
        "Happy": 0,
        "Neutral": 1,
        "Surprise": 0,
        "Embarrassment": 0
      },
      "dominant_emotion": "Neutral",
      "confidence": 0.87
    }
  ],
  "processing_time": 0.15
}
```

## 🏗️ Архитектура системы

### Модели и технологии

- **Детекция лиц**: YOLO (Ultralytics)
- **Классификация эмоций**: ResNet18 с дообучением
- **Фреймворк**: PyTorch
- **Обработка изображений**: OpenCV

### Характеристики модели

- **Количество классов**: 4
- **Input size**: 224×224 пикселей
- **Формат вывода**: JSON с детализированной информацией
- **Поддержка**: Multiple faces detection

## 📁 Структура проекта

```
emotion-recognition/
├── models/
│   ├── emotion_resnet18.pth
│   └── yolo_face_detector.pt
├── src/
│   ├── face_detector.py      # YOLO детекция лиц
│   ├── emotion_classifier.py # ResNet18 классификатор
│   ├── emotion_detector.py   # Основной класс системы
│   └── utils/
│       ├── preprocess.py
│       └── visualization.py
├── examples/
│   ├── single_face.jpg
│   └── group_photo.jpg
├── results/                  # Папка для результатов
├── requirements.txt
└── README.md
```

## 💻 Использование из командной строки

```bash
# Обработка одного изображения
python src/emotion_detector.py --image path/to/image.jpg --output results/

# Обработка всей папки с изображениями
python src/emotion_detector.py --folder path/to/images/ --output results/

# Использование веб-камеры в реальном времени
python src/emotion_detector.py --webcam
```

## 📊 Метрики производительности

| Метрика | Значение |
|---------|----------|
| **Accuracy** | [Добавьте ваше значение] |
| **Inference Time** | ~150мс на изображение |
| **Поддержка multiple faces** | ✅ Да |
| **Real-time processing** | ✅ Да (зависит от hardware) |

## 🛠️ Установка и настройка

1. **Клонирование репозитория**
```bash
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
```

2. **Установка зависимостей**
```bash
pip install -r requirements.txt
```

3. **Загрузка моделей**
```bash
# Модели должны быть размещены в папке models/
```

## 👥 Пример использования в коде

```python
from emotion_detector import EmotionDetector
import json

# Инициализация
detector = EmotionDetector()

# Обработка изображения
result = detector.process_image("test_image.jpg")

# Сохранение результатов
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

# Визуализация результатов
detector.visualize_result("test_image.jpg", result, "output.jpg")
```

## 🎯 Возможности

- ✅ Детекция нескольких лиц на изображении
- ✅ Высокая точность распознавания эмоций
- ✅ Режим реального времени
- ✅ Экспорт результатов в JSON
- ✅ Поддержка изображений и видео потоков

## 🤝 Разработка

Для дообучения модели или модификации:

```bash
# Дообучение модели эмоций
python src/train_emotion.py --data path/to/dataset/ --epochs 50
```

## 👥 Авторы

- [Ваше имя/команда] - [Ваши контакты]


---

<div align="center">

**⭐ Если проект полезен - поставьте звезду репозиторию!**

</div>
```
