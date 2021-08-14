import boto3
import os
import json
import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.cluster import DBSCAN
import time
from botocore import config


solution_identifier = {"user_agent_extra": "AwsSolution/SO8016/1.0.0"}
config = config.Config(**solution_identifier)

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

    request = json.loads(event['body'])
    activity_id = request.get('activity_id', None)
    print('activity_id = {}'.format(activity_id))

    # --------------------------------------------------------------------------------- #
    # ------             Query All Faces with Given Activity ID                -------- #
    # --------------------------------------------------------------------------------- #
    faces = query_all_faces_in_dynamodb(
        dynamodb_table_name=dynamodb_table_name,
        activity_id=activity_id
    )

    # --------------------------------------------------------------------------------- #
    # ------           Generate Faces Association Summary Report               -------- #
    # --------------------------------------------------------------------------------- #
    if len(faces) == 0:
        ret_body = "faces library empty"
    else:
        print('Len(faces) = {}'.format(len(faces)))
        print('faces[0] = {}'.format(faces[0]))
        print('type(faces[0]) = {}'.format(type(faces[0])))

        t_cluster_start = time.time()
        face_feats = list()
        face_info = list()
        for index, face in enumerate(faces):
            face_feats.append(face['representation'])
            face_info.append(
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

        face_feats = np.array(face_feats)
        print('face_feats.shape = {}'.format(face_feats.shape))

        # clustering based on cosine distance
        cosine_cluster = DBSCAN(metric='cosine', eps=FACE_RECOGNITION_THRESHOLD['cosine'], min_samples=1)
        cosine_cluster_labels = cosine_cluster.fit_predict(face_feats)
        print('cosine_cluster_labels = {}'.format(cosine_cluster_labels))
        print('len(list(set(cosine_cluster_labels))) = {}'.format(len(list(set(cosine_cluster_labels)))))
        t_cluster_end = time.time()
        print('Clustering Time Cost = {}'.format(1000.0 * (t_cluster_end - t_cluster_start)))

        # # clustering based on euclidean distance
        # norm_vector = np.linalg.norm(face_feats, axis=-1)
        # norm_vector = np.expand_dims(norm_vector, axis=-1)
        # faces_feats_norm = face_feats / norm_vector
        # euclidean_cluster = DBSCAN(metric='euclidean', eps=FACE_RECOGNITION_THRESHOLD['euclidean_l2'], min_samples=1)
        # euclidean_cluster_labels = euclidean_cluster.fit_predict(faces_feats_norm)

        faces_cluster = dict()

        for index, label in enumerate(cosine_cluster_labels):
            if label in faces_cluster.keys():
                faces_cluster[label].append(face_info[index])
            else:
                faces_cluster[label] = [face_info[index]]

        ret_body = json.dumps(faces_cluster)

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


