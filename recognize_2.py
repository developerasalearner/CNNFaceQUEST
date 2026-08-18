import urllib
import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"/Users/chinmayakumarpalo/Desktop/CNNFaceQUEST/haarcascade_frontalface_default.xml")

model = load_model(r"/Users/chinmayakumarpalo/Desktop/CNNFaceQUEST/Final_model_10.h5")

URL = 'http://100.72.100.103:8080/shot.jpg'

def get_pred_label(pred):
    labels = ['CHIKU', 'CHINU', 'MAMA']
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255
    return img


ret = True
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

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()