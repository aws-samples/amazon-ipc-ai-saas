import boto3
import os
import json
import base64
import time
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

s3 = boto3.client('s3', config=config)


def handler(event, context):
    image_assets_bucket_name = os.environ['IMAGE_ASSETS_BUCKET_NAME']
    print('Image Assets Bucket Name = {}'.format(image_assets_bucket_name))

    request = json.loads(event['body'])
    activity_id = request.get('activity_id', None)
    image_id = request.get('image_id', None)
    image_base64_enc = request.get('image_base64_enc', None)

    ret_body = 'success'
    if activity_id is None:
        ret_body = "activity_id null error"
    if image_id is None:
        ret_body = "image_id null error"
    if image_base64_enc is None:
        ret_body = "image_base64_enc null error"

    if activity_id is not None and image_id is not None and image_base64_enc is not None:
        print('Activity Id = {}'.format(activity_id))
        print('Image Id = {}'.format(image_id))

        image_data = base64.b64decode(image_base64_enc)
        s3_put_response = s3.put_object(
            Bucket=image_assets_bucket_name,
            Key=os.path.join(activity_id, image_id + '.png'),
            Body=image_data)

        try:
            http_status_code = s3_put_response["ResponseMetadata"]["HTTPStatusCode"]
            if http_status_code != 200:
                ret_body = "failed to save into S3 bucket"
            else:
                print('Successfully save image into S3 bucket.')
        except Exception as e:
            ret_body = "failed to save into S3 bucket"

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
