import pickle
import cv2
import numpy as np
import pyttsx3
import threading
import queue

# инициализация движка
engine = pyttsx3.init()
engine.setProperty('rate', 120)
engine.setProperty('volume', 1.0)

# ставим голос Microsoft David
engine.setProperty('voice', "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0")

labels = {0: "A", 1: "B", 2: "C", 3: "Nothing"}

with open('model.sign.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Камера не открылась")
    exit()

last_letter = None
speech_queue = queue.Queue()

def speech_worker():
    while True:
        letter = speech_queue.get()
        if letter is None:
            break
        engine.say(f"This is letter {letter}")
        engine.runAndWait()
        speech_queue.task_done()

# запускаем поток озвучки
threading.Thread(target=speech_worker, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    y_pred = model.predict(img)
    predicted_class = np.argmax(y_pred)
    predicted_label = labels[predicted_class]

    if predicted_label != last_letter and predicted_label != "Nothing":
        speech_queue.put(predicted_label)
        last_letter = predicted_label

    cv2.putText(frame, f"{predicted_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# останавливаем поток
speech_queue.put(None)
speech_queue.join()

cap.release()
cv2.destroyAllWindows()









 



