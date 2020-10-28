from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
import requests
from random import randint

### Arquivo de input
image_filename = "photo.png"

### Emoções analisadas
emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

### API de remoção de background
#response = requests.post(
#  'https://api.remove.bg/v1.0/removebg',
#  files={'image_file': open(image_filename, 'rb')},
#  data={'size': 'auto'},
#  headers={'X-Api-Key': 'LAHpJwBK3Ly6GsqLuE2tb5Kh'},
#)

#if response.status_code == requests.codes.ok:
#  with open('photo-no-background.png', 'wb') as out:
#    out.write(response.content)
#else:
#    print("Error:", response.status_code, response.text)

# Identificando a localização e selecionando a face
image = face_recognition.load_image_file("./photo-no-background.png")
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]
#image_save = Image.fromarray(face_image)
#image_save.save("face.jpg")

### Analisando a emoção da face
#face_image = cv2.imread("face.jpg")
face_image = cv2.resize(face_image, (48, 48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
model = load_model("./emotion_detector_models/model_v6_23.hdf5")
predicted_class = np.argmax(model.predict(face_image))
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)

# Carregando foto sem o background
img = Image.open("photo-no-background.png")

# Seleciona um background aleatório baseado na emoção identificada
backgroundPath = "./IMG/" + str(predicted_label) + "/0" + str(randint(1,1)) + ".jpg"

# Carrega background
background = Image.open(backgroundPath)

# Redimencionado foto baseado na proporção background/foto
propResize = min(background.size[0]/img.size[0], background.size[1]/img.size[1])
img = img.resize((int(img.size[0]*propResize), int(img.size[1]*propResize)),Image.ANTIALIAS)

# Junção da foto sem o background + background novo
background.paste(img, (int((background.size[0]-img.size[0])*0.25), background.size[1]-img.size[1]), img)
background.save('photo-final.png',"PNG")