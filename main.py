from PIL import Image
import os
import requests
from random import randint
import boto3
import json
import tempfile


# Flags
should_remove_background = False  # Usar com moderação


def get_filename_no_extension(filename):
    return filename.split('.')[-2].split('/')[-1]

# Extrai caracteristicas da face detectada


def detect_faces(photo):
    client = boto3.client('rekognition')

    with open(photo, 'rb') as image:
        response = client.detect_faces(
            Image={'Bytes': image.read()}, Attributes=['ALL'])

    return response['FaceDetails']


# Extrai a emoção da face com maior certeza
def predict_label(image_filename):
    face_details = detect_faces(image_filename)
    image_emotions = face_details[0]["Emotions"]
    return image_emotions[0]["Type"]


# Chama a API para retirar o background
def remove_background(filename):
    filename = get_filename_no_extension(filename) + '-no-background.png'
    if (should_remove_background):
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(image_filename, 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': 'LAHpJwBK3Ly6GsqLuE2tb5Kh'},
        )

        if response.status_code == requests.codes.ok:
            with open(filename, 'wb') as out:
                out.write(response.content)
            image_no_background = response.content
        else:
            print("Error:", response.status_code, response.text)
    return filename


# Redimencionado foto baseado na proporção alvo/fonte
def resize_image(source_filename, target_filename):
    source = Image.open(source_filename)
    target = Image.open(target_filename)
    propResize = min(target.size[0] / source.size[0],
                     target.size[1] / source.size[1])
    resized_photo = source.resize((int(source.size[0] * propResize),
                                   int(source.size[1] * propResize)), Image.ANTIALIAS)
    filename = get_filename_no_extension(source_filename) + '-resized.png'
    resized_photo.save(filename, "PNG")
    return filename


# Junção da foto sem o background + background novo
def paste_image(source_filename, target_filename):
    source = Image.open(source_filename)
    target = Image.open(target_filename)
    target.paste(source, (int(
        (target.size[0] - source.size[0]) * 0.5), target.size[1] - source.size[1]), source)
    filename = get_filename_no_extension(
        source_filename) + '-' + get_filename_no_extension(target_filename) + '.png'
    target.save(filename, "PNG")
    return filename


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Arquivo de input
        image_filename = "photo.jpg"

        image_no_bg_filename = remove_background(image_filename)

        predicted_label = predict_label(image_no_bg_filename)

        # Seleciona um background aleatório baseado na emoção identificada
        background_filename = "./IMG/" + \
            str(predicted_label) + "/0" + str(randint(1, 1)) + ".jpg"

        image_no_bg_resized_filename = resize_image(
            image_no_bg_filename, background_filename)

        paste_image(image_no_bg_resized_filename, background_filename)
