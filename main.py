from PIL import Image
import requests
from random import randint
import boto3
import json


def detect_faces(photo):
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_faces(
            Image={'Bytes': image.read()}, Attributes=['ALL'])

    return response['FaceDetails']


def detect_labels_local_file(photo):
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})

    return response['Labels']


# Arquivo de input
image_filename = "photo.jpg"

# API de remoção de
remove_background = False  # Usar com moderação
if (remove_background):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(image_filename, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'LAHpJwBK3Ly6GsqLuE2tb5Kh'},
    )

    if response.status_code == requests.codes.ok:
        with open('photo-no-background.png', 'wb') as out:
            out.write(response.content)
        image_no_background = response.content
    else:
        print("Error:", response.status_code, response.text)

# Extrai caracteristicas da imagem
#label_count = detect_labels_local_file(image_filename)
#print("Labels detected: " + str(label_count))

# Extrai caracteristicas da face detectada
face_details = detect_faces(image_filename)
#print("Image Details: " + str(face_details))
image_emotions = face_details[0]["Emotions"]
predicted_label = image_emotions[0]["Type"]
print(predicted_label)

# Carregando foto sem o background
image_no_background = Image.open("photo-no-background.png")

# Seleciona um background aleatório baseado na emoção identificada
backgroundPath = "./IMG/" + \
    str(predicted_label) + "/0" + str(randint(1, 1)) + ".jpg"

# Carrega background
background = Image.open(backgroundPath)

# Redimencionado foto baseado na proporção background/foto
propResize = min(background.size[0] / image_no_background.size[0],
                 background.size[1] / image_no_background.size[1])
image_no_background = image_no_background.resize((int(image_no_background.size[0] * propResize),
                                                  int(image_no_background.size[1] * propResize)), Image.ANTIALIAS)

# Junção da foto sem o background + background novo
background.paste(image_no_background, (int(
    (background.size[0] - image_no_background.size[0]) * 0.25), background.size[1] - image_no_background.size[1]), image_no_background)
background.save('photo-final.png', "PNG")
