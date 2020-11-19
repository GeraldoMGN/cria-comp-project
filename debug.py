import lambda_function
import base64
import json

if __name__ == '__main__':
    encoded_string = ""
    with open("test.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    event = {
        "body": str(encoded_string)
    }

    response = lambda_function.lambda_handler(event, None)
    parsed_response = json.loads(response.get("body")).get("img")
    image_string = parsed_response[parsed_response.find('64,') + 3:]

    with open('test_out.jpg', "wb") as fh:
        fh.write(base64.b64decode(image_string))
