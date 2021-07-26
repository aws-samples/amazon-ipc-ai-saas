import boto3
import os
import json
import base64
import time
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

sagemaker_runtime_client = boto3.client('runtime.sagemaker', config=config)
s3 = boto3.client('s3', config=config)


def handler(event, context):
    # parse environment parameters
    sagemaker_endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']
    events_s3_bucket_name = os.environ['EVENTS_S3_BUCKET_NAME']
    request_events_snapshot_enabled = os.environ['REQUEST_EVENTS_SNAPSHOT_ENABLED']

    # obtain image bytes
    request = json.loads(event['body'])
    image_base64_enc = request['image_base64_enc']
    timestamp = request['timestamp']
    request_id = str(request['request_id'])

    # invoke Sagemaker endpoint to predict the skeleton
    sm_endpoint_response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({
            'image_bytes': image_base64_enc
        }),
    )

    detection_str = sm_endpoint_response['Body'].read().decode()

    # return response
    response = {
        'statusCode': 200,
        'body': detection_str,
        "headers":
            {
                "Content-Type": "application/json",
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
            }
    }

    # write input images and response to S3 bucket
    if request_events_snapshot_enabled == 'Yes':
        save_bucket_prefix = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        image_data = base64.b64decode(image_base64_enc)
        dump_input_image_response = s3.put_object(
            Bucket=events_s3_bucket_name,
            Key=os.path.join(save_bucket_prefix, request_id + '.png'),
            Body=image_data)

        dump_detection_result_response = s3.put_object(
            Bucket=events_s3_bucket_name,
            Key=os.path.join(save_bucket_prefix, request_id + '.json'),
            Body=json.dumps(response))

    return response


