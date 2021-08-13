import boto3
import os
import json
import base64
import time
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

sagemaker_runtime_client = boto3.client('runtime.sagemaker', config=config)
dynamodb = boto3.client('dynamodb', config=config)


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


def handler(event, context):
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']
    sagemaker_endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']

    request = json.loads(event['body'])
    activity_id = request.get('activity_id', None)
    image_base64_enc = request.get('image_base64_enc', None)

    ret_body = 'success'

    # invoke SageMaker inference endpoint to perform face detection and representation
    anchor_face = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({
            'image_bytes': image_base64_enc
        }),
    )
    anchor_face = anchor_face['Body'].read().decode()
    print("Faces Response = {}".format(anchor_face))

    # get all faces in given activity_id
    table = dynamodb.Table(dynamodb_table_name)
    query_response = table.query(
        KeyConditionExpression=Key('activity_id').eq(activity_id)
    )
    faces_pool = query_response['Items']
    print('Faces = {}'.format(faces_pool))
    print(type(faces_pool))

    # TODO: compare anchor_face with faces pool

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


