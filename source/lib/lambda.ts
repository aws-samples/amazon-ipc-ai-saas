import { Construct } from "@aws-cdk/core";
import * as iam from "@aws-cdk/aws-iam";
import * as lambda from "@aws-cdk/aws-lambda";
import * as cdk from "@aws-cdk/core";
import {S3EventSource} from "@aws-cdk/aws-lambda-event-sources";
import * as s3 from '@aws-cdk/aws-s3';
import {Bucket} from '@aws-cdk/aws-s3';
import {Function} from "@aws-cdk/aws-lambda";


export interface LambdaHandlersProps {
    readonly imageAssets: Bucket,
    readonly facesTableName: string,
    readonly sagemakerInferenceEndpointName: string,
}


export class LambdaHandlers extends Construct {
    public readonly imageUploader: Function;
    public readonly faceDetectorAndRepresenter: Function;
    public readonly activitySummary: Function;
    public readonly faceQuery: Function;


    constructor(scope: Construct, id: string, props: LambdaHandlersProps) {
        super(scope, id);


        /**
         * Create Role for Lambda Function
         */
        const faceRecognitionLambdaRole = new iam.Role(
        this,
        'faceRecognitionLambdaRole',
        {
            assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonDynamoDBFullAccess')
            ]
        });


        /**
         * Create Lambda Function: HTTPS POST Request with Image Base64 Data as Payload to
         * Invoke Face Detection & Representation Inference
         */
         this.imageUploader = new lambda.Function(
            this,
            'imageUploader',
            {
                code: new lambda.AssetCode('lambda/image_uploader'),
                handler: 'main.handler',
                runtime: lambda.Runtime.PYTHON_3_6,
                environment: {
                    IMAGE_ASSETS_BUCKET_NAME: props.imageAssets.bucketName,
                },
                timeout: cdk.Duration.minutes(10),
                role: faceRecognitionLambdaRole,
                memorySize: 1024,
            }
        );


        /**
         * Create Lambda Function: Image Uploading Event will Trigger this Function to Invoke Face Detection
         * and Features Representation. Finally the Face Embedding Feature Vector will Writen into DynamoDB
         */
        this.faceDetectorAndRepresenter = new lambda.Function(
            this,
            'faceDetectorAndRepresenter',
            {
                code: new lambda.AssetCode('lambda/face_detect_represent'),
                handler: 'main.handler',
                runtime: lambda.Runtime.PYTHON_3_6,
                environment: {
                    DYNAMODB_TABLE_NAME: props.facesTableName,
                    IMAGE_ASSETS_BUCKET_NAME: props.imageAssets.bucketName,
                    SAGEMAKER_ENDPOINT_NAME: props.sagemakerInferenceEndpointName,
                },
                timeout: cdk.Duration.minutes(10),
                role: faceRecognitionLambdaRole,
                memorySize: 8196,
            }
        );

        this.faceDetectorAndRepresenter.addEventSource(new S3EventSource( props.imageAssets,
            { events: [ s3.EventType.OBJECT_CREATED],  filters: [ { suffix: '.png' } ] } ));


        /**
         * Create Lambda Function: User Can Request the All Summary Info Based on Given Activity Id, it will
         * Query the DynamoDB Table to Return all Records
         */
        this.activitySummary = new lambda.Function(
            this,
            'activitySummary',
            {
                code: lambda.Code.fromAsset(
                    './lambda/activity_summary/',
                    {
                        bundling: {
                            image: lambda.Runtime.PYTHON_3_6.bundlingImage,
                            command: [
                                'bash', '-c',
                                'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
                            ]
                        }
                    }
                ),
                handler: 'main.handler',
                runtime: lambda.Runtime.PYTHON_3_6,
                environment: {
                    DYNAMODB_TABLE_NAME: props.facesTableName,
                },
                role: faceRecognitionLambdaRole,
                timeout: cdk.Duration.minutes(10),
                memorySize: 512,
            }
        );


        /**
         * Create Lambda Function: User also can query the dynamodb table based on a given image, it will invoke
         * the face detection and representation service (SageMaker Endpoint) and then compare the face embedding
         * vector with all faces in DynamoDB, finally return all associated iamges with face detection info
         */
        this.faceQuery = new lambda.Function(
            this,
            'faceQuery',
            {
                code: lambda.Code.fromAsset(
                    './lambda/face_query/',
                    {
                        bundling: {
                            image: lambda.Runtime.PYTHON_3_6.bundlingImage,
                            command: [
                                'bash', '-c',
                                'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
                            ]
                        }
                    }
                ),
                handler: 'main.handler',
                runtime: lambda.Runtime.PYTHON_3_6,
                environment: {
                    SAGEMAKER_ENDPOINT_NAME: props.sagemakerInferenceEndpointName,
                    DYNAMODB_TABLE_NAME: props.facesTableName,
                },
                timeout: cdk.Duration.minutes(10),
                role: faceRecognitionLambdaRole,
                memorySize: 8196,
            }
        );


    }
}




