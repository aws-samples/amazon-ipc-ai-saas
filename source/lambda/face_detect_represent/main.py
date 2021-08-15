import boto3
import os
import json
import base64
import uuid
import time
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

sagemaker_runtime_client = boto3.client('runtime.sagemaker', config=config)
s3 = boto3.client('s3', config=config)
dynamodb = boto3.client('dynamodb', config=config)


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


def handler(event, context):
    ret_body = 'success'
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']
    image_assets_bucket_name = os.environ['IMAGE_ASSETS_BUCKET_NAME']
    sagemaker_endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']

    # -------------------------------------------------------------------------------------------- #
    # ------              Step 1: Download Image From S3 Bucket to Local Path             -------- #
    # -------------------------------------------------------------------------------------------- #
    s3_key = event['Records'][0]['s3']['object']['key']
    activity_id = s3_key.split('/')[0]
    image_name = s3_key.split('/')[1]
    image_id = image_name.split('.png')[0]
    print('activity_id = {}'.format(activity_id))
    print('image_name = {}'.format(image_name))
    print('image_id = {}'.format(image_id))

    local_image_path = '/tmp/{}'.format(image_name)
    with open(local_image_path, 'wb') as wf:
        s3.download_fileobj(image_assets_bucket_name, s3_key, wf)
    print('Successfully download {} to {}'.format(image_name, local_image_path))

    # -------------------------------------------------------------------------------------------- #
    # ------                 Step 2: Faces Detection & Representation                     -------- #
    # -------------------------------------------------------------------------------------------- #
    all_faces_in_image = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({
            'image_bytes': get_base64_encoding(local_image_path)
        }),
    )
    all_faces_str = all_faces_in_image['Body'].read().decode()
    print("Faces Response = {}".format(all_faces_str))

    # -------------------------------------------------------------------------------------------- #
    # ------                    Step 3: Write Each Face into DynamoDB Table               -------- #
    # -------------------------------------------------------------------------------------------- #
    all_faces = json.loads(all_faces_str)
    image_height = all_faces.get("image_height", -1)
    image_width = all_faces.get("image_width", -1)
    image_channels = all_faces.get("image_channels", -1)
    faces = all_faces.get("faces", list())

    for index, detected_face in enumerate(faces):
        bbox = detected_face['bbox']
        confidence = detected_face['confidence']
        landmarks = detected_face['landmarks']
        representation = detected_face['representation']

        face_record = {
            "activity_id": {'S': activity_id},
            "image_id": {'S': image_id},
            "face_id": {'S': str(uuid.uuid4())},
            "image_width": {'N': str(image_width)},
            "image_height": {'N': str(image_height)},
            "image_channels": {'N': str(image_channels)},
            "bbox": {'S': json.dumps(bbox)},
            "confidence": {'N': str(confidence)},
            "landmarks": {'S': json.dumps(landmarks)},
            "representation": {'S': json.dumps(representation)}
        }

        # write into DynamoDB table
        db_write_response = dynamodb.put_item(
            TableName=dynamodb_table_name,
            Item=face_record
        )

        print("Write face {} / {} into DynamoDB table {}...".format(index + 1, len(faces), dynamodb_table_name))

    # return response
    response = {
        'statusCode': 200,
        'body': ret_body,
        "headers":
            {
                "Content-Type": "application/json",
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
            }
    }

    return response


