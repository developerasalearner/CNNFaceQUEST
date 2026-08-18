import urllib
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"D:\FACE RECOGNISITION\FACE RECOGNISITION\haarcascade_frontalface_default (1).xml")

model = load_model(r"D:\FACE RECOGNISITION\Minortwo_Project_final_Data.h5")

URL = 'http://192.0.0.4:8080/shot.jpg'

labels = ['Aniket', 'Chinmaya', 'Kausik', 'Suryapratap', 'Uttam']
attendance = {}

def get_pred_label(pred):
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255
    return img

def mark_attendance(name):
    date_string = datetime.now().strftime("%Y-%m-%d")
    time_string = datetime.now().strftime("%H:%M:%S")
    if name in attendance:
        attendance[name].append(time_string)
    else:
        attendance[name] = [time_string]
    print(f"Attendance marked for {name} at {time_string}")

def save_attendance():
    df = pd.DataFrame(attendance)
    df.to_excel('attendance.xlsx', index=False)
    print("Attendance saved to attendance.xlsx")

def start_program():
    ret = True
    blink_counter = {}
    start_time = None

    while ret:
        img_url = urllib.request.urlopen(URL)
        image = np.array(bytearray(img_url.read()), np.uint8)
        frame = cv2.imdecode(image, -1)

        faces = classifier.detectMultiScale(frame, 1.5, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            text = get_pred_label(np.argmax(model.predict(preprocess(face))))
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            text_x = x + (w - text_size[0]) // 2
            text_y = y + h + text_size[1] + 10 + i * (text_size[1] + 10)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            if text not in blink_counter:
                blink_counter[text] = 0

            if blink_counter[text] >= 10:
                if text not in attendance:
                    mark_attendance(text)

            if text == get_pred_label(np.argmax(model.predict(preprocess(face)))):
                blink_counter[text] += 1
            else:
                blink_counter[text] = 0

        cv2.imshow("capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def trigger_button():
    global attendance
    attendance = {}
    trigger_pressed = False

    while True:
        user_input = input("Press 't' to toggle the program, 'e' to save attendance, or 'q' to quit: ")
        if user_input == 't':
            trigger_pressed = not trigger_pressed
            if trigger_pressed:
                print("Program started.")
                start_program()
            else:
                print("Program stopped.")
        elif user_input == 'e':
            save_attendance()
        elif user_input == 'q':
            break
        else:
            print("Invalid input. Please try again.")

trigger_button()