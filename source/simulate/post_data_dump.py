import base64
import json
import time

def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc

image_base64_enc = get_base64_encoding(full_path='./test_imgs/persons/persons.jpg')

# Step 2: send request to backend
request_body = {
    "timestamp": str(time.time()),
    "request_id": 1242322,
    "image_base64_enc": image_base64_enc
}

json.dump(request_body, open('post_data.txt', 'w'))
