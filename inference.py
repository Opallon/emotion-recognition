import cv2
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import json
import time, math

# ------------------------
# ГЛОБАЛЬНЫЕ СПИСКИ И ПЕРЕМЕННЫЕ ДЛЯ ОТСЛЕЖИВАНИЯ
# ------------------------
resnet_predictions = []  # Для "среднего класса" (последние N предсказаний)
repeat_count = 0
current_repeated_class = None
last_written_class = None

# ------------------------
# ФУНКЦИЯ ДЛЯ ДОБАВЛЕНИЯ ЗНАЧЕНИЙ В JSON-ФАЙЛ (как список)
# ------------------------


def append_to_json_list(filename, mode, value):
    """
    Считывает массив из JSON-файла filename (если нет — создаёт пустой),
    добавляет в него новое значение и перезаписывает обратно как валидный JSON.
    """
    """
    Инкрементирует счетчик для заданного класса в JSON-файле.

    Args:
        filename (str): Путь к JSON-файлу.
        class_index (int): Индекс класса (0: Happy, 1: Neutral, 2: Surprise, 3: Embarrassment).
    """

    # Инициализация словаря с классами
    class_mapping = {
        "Mode": "100",
        "Smile": "0",
        "Neutral": "1",
        "Surprise": "2",
        "Embarrassment": "3",
        "CurrentGesture": "221",
    }
    label = ["Smile", "Neutral", "Surprise", "Embarrassment", "CurrentGesture"]
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
            if not isinstance(data_dict, dict):
                data_dict = {
                    k: mode if k == "Mode" else 0 for k in class_mapping.keys()
                }
    except (FileNotFoundError, json.JSONDecodeError):
        data_dict = {k: mode if k == "Mode" else 0 for k in class_mapping.keys()}
    if mode == 1:
        # Проверка валидности класса
        if label[value] in class_mapping:
            print(data_dict[label[value]])
            data_dict[label[value]] += 1
        else:
            print(f"Неизвестный индекс класса: {value}")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
    elif mode == 2:
        try:
            if "CurrentGesture" in class_mapping:
                print(data_dict["CurrentGesture"])
                data_dict["CurrentGesture"] = value
            else:
                print(f"Неизвестный индекс класса: {value}")

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
        except:
            data_dict = {k: mode if k == "Mode" else 0 for k in class_mapping.keys()}
            # print(data_dict)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)


def detect_and_crop(image, model, confidence_threshold=0.5):
    """
    Оптимизированная версия функции детекции и обрезки
    """
    try:
        results = model(image, conf=confidence_threshold, stream=True, verbose=False)
        cropped_images = []

        for r in results:
            for box in r.boxes:
                b = box.xyxy[0]
                c = box.cls
                x1, y1, x2, y2 = map(int, b)
                class_name = model.names[int(c)]

                cropped_img = image[y1:y2, x1:x2].copy()
                cropped_images.append((cropped_img, class_name))

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                y_text = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(
                    image,
                    class_name,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        return cropped_images

    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def resnet18_inference(img, model, transform):
    """
    Оптимизированная версия функции инференса
    """
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_t = transform(img_resized)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = model(batch_t)  # shape [1, num_classes]
            probabilities = torch.nn.functional.softmax(out[0], dim=0)

        return probabilities

    except Exception as e:
        print(f"Ошибка при инференсе: {e}")
        return None


class handDetector:
    def __init__(
        self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    img,
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[2] + 20, bbox[3] + 20),
                    (0, 255, 0),
                    2,
                )
        return self.lmList, bbox

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) < max(p1, p2) + 1:
            return float("inf"), img, []

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        if len(self.lmList) < max(self.tipIds) + 1:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


# ------------------------
# ОТДЕЛЬНЫЕ БЛОКИ ЛОГИКИ
# ------------------------


def process_face(frame, model_yolo, model_resnet, transform, hands, mode):
    """
    Функция обработки кадра для YOLO (лицо) + ResNet (классификация).
    Доп. проверка на наличие руки (MediaPipe).
    Если рук нет -> ResNet. Если 5 раз подряд один класс (!= 1 и != last_written_class)
    -> пишем этот класс в 'expected.json' (добавляем в массив).
    Если рука обнаружена -> пишем "3" (добавляем в массив).
    """
    global resnet_predictions
    global repeat_count, current_repeated_class, last_written_class

    cropped_images = detect_and_crop(frame, model_yolo, 0.25)

    if cropped_images:
        for cropped_img, class_name in cropped_images:
            imgRGB = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(imgRGB)

            # Если РУКИ обнаружены => записываем 3 в expected.json
            if hand_results.multi_hand_landmarks is not None:
                # Чтобы не спамить 3, проверяем, записана ли она последней
                if last_written_class != 3:
                    append_to_json_list("expected.json", mode, 3)
                    last_written_class = 3
                    print("Рука обнаружена: в expected.json добавлен класс 3.")

                # Сбрасываем счётчики для повторов ResNet
                current_repeated_class = None
                repeat_count = 0

            else:
                # Если руки НЕТ, делаем инференс ResNet
                probabilities = resnet18_inference(cropped_img, model_resnet, transform)
                if probabilities is not None:
                    _, predicted_class = torch.max(probabilities, 0)
                    predicted_class_value = predicted_class.item()

                    # Сохраняем предсказание в список для "среднего класса" (по желанию)
                    resnet_predictions.append(predicted_class_value)
                    if len(resnet_predictions) > 10:
                        resnet_predictions.pop(0)
                    avg_class = int(
                        round(sum(resnet_predictions) / len(resnet_predictions))
                    )
                    print(
                        f"Текущий класс: {predicted_class_value}, Средний класс (10 предсказаний): {avg_class}"
                    )

                    # Логика "5 раз подряд" => записать класс
                    if avg_class != current_repeated_class:
                        current_repeated_class = avg_class
                        repeat_count = 1
                    else:
                        repeat_count += 1

                    if repeat_count == 5:
                        # Класс != 1, != last_written_class => записать
                        if avg_class != 1 and avg_class != last_written_class:
                            append_to_json_list("expected.json", mode, avg_class)
                            last_written_class = avg_class
                            print(
                                f"В expected.json добавлен класс {avg_class} (5 повторов)."
                            )

                        # Сброс счётчика
                        repeat_count = 0
                else:
                    print("Ошибка при инференсе ResNet")


def read_max_from_json(filename):
    """Читает текущий максимум из JSON-файла. Возвращает 0 если файл невалиден."""
    try:
        with open(filename, "r") as f:
            content = json.load(f)
            print(content)
            if isinstance(content["CurrentGesture"], (int, float)):
                return int(content["CurrentGesture"])
            return 0
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return 0


# def write_max_to_json(filename, value):
#    """Записывает значение как целое число в JSON-файл."""
#    with open(filename, "w") as f:
#        json.dump("CurrentGesture": int(value), f)


def process_hand(frame, detector, mode):
    """
    Обрабатывает кадр для распознавания жестов:
    1. Находит положение руки и пальцев
    2. Считает сумму поднятых пальцев
    3. Обновляет predict.json только если:
       - Новое значение > текущего максимума
       - Значение не равно нулю
    """
    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame, draw=False)

    if len(lmList) == 0:
        return  # Не делаем ничего если рука не обнаружена

    fingers_state = detector.fingersUp()
    fingers_sum = sum(fingers_state)

    if fingers_sum == 0:
        return  # Не записываем нули

    # Получаем текущий максимум
    try:
        current_max = read_max_from_json("expected.json")
    except:
        append_to_json_list("expected.json", mode, 0)
        current_max = 0

    if fingers_sum > current_max:
        append_to_json_list("expected.json", mode, fingers_sum)
        print(f"Обновлён максимум: {fingers_sum} (предыдущий: {current_max})")
