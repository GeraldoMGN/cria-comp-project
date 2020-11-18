from PIL import Image
import os
import requests
from random import randint
import boto3
import json
import tempfile
import base64


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
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(filename, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'gy6f2aUkPkXLLrmDbhdgmMo1'},
    )

    if response.status_code == requests.codes.ok:
        result_filename = '/tmp/' + get_filename_no_extension(
            filename) + '-no-background.png'
        with open(result_filename, 'wb') as out:
            out.write(response.content)
        return result_filename
    else:
        print("Error:", response.status_code, response.text)
    return None


# Redimencionado foto baseado na proporção alvo/fonte
def resize_image(source_filename, target_filename):
    source = Image.open(source_filename)
    target = Image.open(target_filename)
    propResize = min(target.size[0] / source.size[0],
                     target.size[1] / source.size[1])
    resized_photo = source.resize((int(source.size[0] * propResize),
                                   int(source.size[1] * propResize)), Image.ANTIALIAS)
    filename = '/tmp/' + \
        get_filename_no_extension(source_filename) + '-resized.png'
    resized_photo.save(filename, "PNG")
    return filename


# Junção da foto sem o background + background novo
def paste_image(source_filename, target_filename):
    source = Image.open(source_filename)
    target = Image.open(target_filename)
    target.paste(source, (int(
        (target.size[0] - source.size[0]) * 0.5), target.size[1] - source.size[1]), source)
    filename = '/tmp/' + get_filename_no_extension(
        source_filename) + '-' + get_filename_no_extension(target_filename) + '.png'
    target.save(filename, "PNG")
    return filename


def lambda_handler(event, context):
    event_body = event.get('body')
    image_string = event_body[event_body.find('/9'):]

    image_filename = "photo.jpg"

    with open('/tmp/' + image_filename, "wb") as fh:
        fh.write(base64.b64decode(image_string))

    image_no_bg_filename = remove_background('/tmp/' + image_filename)

    predicted_label = predict_label(image_no_bg_filename)

    # Seleciona um background aleatório baseado na emoção identificada
    background_filename = "IMG/" + \
        str(predicted_label) + "/0" + str(randint(1, 3)) + ".jpg"

    image_no_bg_resized_filename = resize_image(
        image_no_bg_filename, background_filename)

    result = paste_image(image_no_bg_resized_filename, background_filename)

    with open(result, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        response_body = {
            'img': 'data:image/jpeg;base64,' + encoded_image
        }

        return {
            'statusCode': 200,
            'body': json.dumps(response_body),
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'PUT,POST,GET'
            },
        }
