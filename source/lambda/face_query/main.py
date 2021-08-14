import boto3
import os
import json
import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

sagemaker_runtime_client = boto3.client('runtime.sagemaker', config=config)
dynamodb = boto3.client('dynamodb', config=config)


# DO NOT MODIFY
FACE_RECOGNITION_THRESHOLD = {
    'cosine': 0.68,
    'euclidean_l2': 1.13
}


def query_all_faces_in_dynamodb(dynamodb_table_name, activity_id):
    t_query_start = time.time()
    query_response = dynamodb.query(
        TableName=dynamodb_table_name,
        Limit=10000,
        KeyConditionExpression="activity_id = :v1",
        ExpressionAttributeValues={
            ':v1': {
                'S': activity_id,
            },
        },
    )

    last_evaluated_key = query_response.get('LastEvaluatedKey', None)
    faces = query_response['Items']

    while last_evaluated_key:
        query_response = dynamodb.query(
            TableName=dynamodb_table_name,
            Limit=10000,
            KeyConditionExpression="activity_id = :v1",
            ExpressionAttributeValues={
                ':v1': {
                    'S': activity_id,
                },
            },
            ExclusiveStartKey=last_evaluated_key
        )

        last_evaluated_key = query_response.get('LastEvaluatedKey', None)
        faces.extend(query_response['Items'])

    t_query_end = time.time()
    print('Query {} Face Records Cost {} ms'.format(len(faces), 1000.0 * (t_query_end - t_query_start)))
    return faces


def handler(event, context):
    dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']
    sagemaker_endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']

    request = json.loads(event['body'])
    activity_id = request.get('activity_id', None)
    image_base64_enc = request.get('image_base64_enc', None)

    # -------------------------------------------------------------------------------------------- #
    # ------                 Step 1: Faces Detection & Representation                     -------- #
    # -------------------------------------------------------------------------------------------- #
    detected_faces = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({
            'image_bytes': image_base64_enc
        }),
    )
    faces_response = detected_faces['Body'].read().decode()
    print("Faces Response = {}".format(faces_response))
    all_faces = json.loads(faces_response)
    anchor_faces = all_faces["faces"]
    if len(anchor_faces) == 0:
        ret_body = "no face detected in query image"
    else:
        anchor_face_representation = np.array(anchor_faces[0]["representation"])
        anchor_face_feats = np.expand_dims(anchor_face_representation, axis=0)

        # -------------------------------------------------------------------------------------------- #
        # ------               Step 2: Query All Faces with Given Activity ID                 -------- #
        # -------------------------------------------------------------------------------------------- #
        target_faces = query_all_faces_in_dynamodb(
            dynamodb_table_name=dynamodb_table_name,
            activity_id=activity_id
        )

        # -------------------------------------------------------------------------------------------- #
        # ------                     Step 3: Face Distance Calculation                        -------- #
        # -------------------------------------------------------------------------------------------- #
        if len(target_faces) == 0:
            ret_body = "faces library empty"
        else:
            target_face_feats = list()
            target_face_info = list()
            for _, face in enumerate(target_faces):
                target_face_feats.append(face['representation'])
                target_face_info.append(
                    {
                        'activity_id': face['activity_id'],
                        'image_id': face['image_id'],
                        'face_id': face['face_id'],
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'landmarks': {
                            "left_eye": face['left_eye'],
                            "right_eye": face['right_eye'],
                            "nose": face['nose'],
                            "left_mouth": face['left_mouth'],
                            "right_mouth": face['right_mouth']
                        },
                    }
                )

            target_face_feats = np.array(target_face_feats)
            print('target_face_feats.shape = {}'.format(target_face_feats.shape))

            cosine_distance_list = cosine_distances(X=anchor_face_feats, Y=target_face_feats)
            cosine_distance_list = np.squeeze(cosine_distance_list)

            # target_norm_vector = np.linalg.norm(target_face_feats, axis=-1)
            # target_norm_vector = np.expand_dims(target_norm_vector, axis=-1)
            # target_face_feats_norm = target_face_feats / target_norm_vector
            # anchor_norm_vector = np.linalg.norm(anchor_face_feats, axis=-1)
            # anchor_norm_vector = np.expand_dims(anchor_norm_vector, axis=-1)
            # anchor_face_feats_norm = anchor_face_feats / anchor_norm_vector
            #
            # euclidean_distance_list = euclidean_distances(X=anchor_face_feats_norm, Y=target_face_feats_norm)
            # euclidean_distance_list = np.squeeze(euclidean_distance_list)

            indices = np.where(cosine_distance_list < FACE_RECOGNITION_THRESHOLD['cosine'])[0]

            cluster = list()
            for index in indices:
                cluster.append(target_face_info[index])

            ret_body = json.dumps(cluster)

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


