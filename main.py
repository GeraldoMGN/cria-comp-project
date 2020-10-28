from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
import requests

image = face_recognition.load_image_file("./gabs.jpg")
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[1]
face_image = image[top:bottom, left:right]
image_save = Image.fromarray(face_image)
image_save.save("face.jpg")

emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_image = cv2.imread("./face.jpg")
face_image = cv2.resize(face_image, (48, 48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
model = load_model("./emotion_detector_models/model_v6_23.hdf5")
predicted_class = np.argmax(model.predict(face_image))
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)

#response = requests.post(
#  'https://api.remove.bg/v1.0/removebg',
#  files={'image_file': open('./gabs.jpg', 'rb')},
#  data={'size': 'auto'},
#  headers={'X-Api-Key': 'LAHpJwBK3Ly6GsqLuE2tb5Kh'},
#)
#if response.status_code == requests.codes.ok:
#  with open('no-bg.png', 'wb') as out:
#    out.write(response.content)
#else:
#    print("Error:", response.status_code, response.text)

img = Image.open("no-bg.png")

background = Image.open("show_1.jpg")

background.paste(img, (0, 0), img)
background.save('final.png',"PNG")