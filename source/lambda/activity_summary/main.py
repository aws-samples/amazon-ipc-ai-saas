import boto3
import os
import json
import base64
import time
from botocore import config
from boto3.dynamodb.conditions import Key


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

dynamodb = boto3.client('dynamodb', config=config)


def handler(event, context):
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']

    request = json.loads(event['body'])
    activity_id = request.get('activity_id', None)

    # query all faces from the DynamoDB table with given activity_id
    ret_body = 'success'
    table = dynamodb.Table(dynamodb_table_name)
    query_response = table.query(
        KeyConditionExpression=Key('activity_id').eq(activity_id)
    )
    faces = query_response['Items']
    print('Faces = {}'.format(faces))
    print(type(faces))

    # TODO: generate faces association relationship and generate the summary json

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


