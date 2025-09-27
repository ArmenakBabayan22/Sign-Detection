import pickle
import cv2
import numpy as np
import pyttsx3
import threading
import queue

# ======================
# Инициализация TTS
# ======================
engine = pyttsx3.init()
engine.setProperty('rate', 120)
engine.setProperty('volume', 1.0)
engine.setProperty('voice', "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0")

speech_queue = queue.Queue()
def speech_worker():
    while True:
        letter = speech_queue.get()
        if letter is None:
            break
        engine.say(f"This is letter {letter}")
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# ======================
# Загрузка модели жестов руки (pickle)
# ======================
labels = {0: "A", 1: "B", 2: "C", 3: "Nothing"}
with open('model.sign.pkl', 'rb') as f:
    gesture_model = pickle.load(f)

# ======================
# Камера
# ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Камера не открылась")
    exit()

last_letter = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------
    # 1) Выделяем руку через цветовую маску (HSV)
    # ----------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Подбираем диапазон кожи (можно корректировать под свет)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Находим контуры руки
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Берем самый большой контур (рука)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # фильтр по размеру
            x, y, w, h = cv2.boundingRect(max_contour)
            hand_crop = frame[y:y+h, x:x+w]

            # ----------------------
            # 2) Предсказание жеста
            # ----------------------
            img = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            y_pred = gesture_model.predict(img)
            predicted_class = np.argmax(y_pred)
            predicted_label = labels[predicted_class]

            if predicted_label != last_letter and predicted_label != "Nothing":
                speech_queue.put(predicted_label)
                last_letter = predicted_label

            # Отображаем рамку и букву
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_label}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# Остановка потока и освобождение камеры
# ======================
speech_queue.put(None)
speech_queue.join()
cap.release()
cv2.destroyAllWindows()

