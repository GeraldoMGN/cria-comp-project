from PIL import Image
import os
import requests
from random import randint
import boto3
import json
import tempfile
import base64
import skimage
from skimage import io, filters
# from skimage.viewer import ImageViewer
import numpy as np


def split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel


def merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=2)

def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian_filter(image, sigma=10, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper


def channel_adjust(channel, values):
    # preserve the original size, so we can reconstruct at the end
    orig_size = channel.shape
    # flatten the image into a single array
    flat_channel = channel.flatten()

    # this magical numpy function takes the values in flat_channel
    # and maps it from its range in [0, 1] to its new squeezed and
    # stretched range
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)

    # put back into the original image shape
    return adjusted.reshape(orig_size)

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
        headers={'X-Api-Key': 'FL7YMvV5rm66HsAmjyS6QhJr'},
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
    original_image = skimage.io.imread(filename)
    original_image = skimage.util.img_as_float(original_image)
    r, g, b = split_image_into_channels(original_image)
    im = merge_channels(r, g, b)
    # 1. Colour channel adjustment example
    r, g, b = split_image_into_channels(original_image)
    r_interp = channel_adjust(r, [0, 0.8, 1.0])
    red_channel_adj = merge_channels(r_interp, g, b)

    # 2. Mid tone colour boost
    r, g, b = split_image_into_channels(original_image)
    r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    r_boost_img = merge_channels(r_boost_lower, g, b)

    # 3. Making the blacks bluer
    bluer_blacks = merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0, 1.0))

    # 4. Sharpening the image
    sharper = sharpen(bluer_blacks, 1.3, 0.3)

    # 5. Blue channel boost in lower-mids, decrease in upper-mids
    r, g, b = split_image_into_channels(sharper)
    b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
    gotham = merge_channels(r, g, b_adjusted)
    skimage.io.imsave('images/5_blue_adjusted.jpg', gotham)
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
