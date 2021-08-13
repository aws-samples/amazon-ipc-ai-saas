import * as cdk from '@aws-cdk/core';
import * as dynamodb from '@aws-cdk/aws-dynamodb';
import * as agw from '@aws-cdk/aws-apigateway';
import * as s3 from '@aws-cdk/aws-s3';
import {SageMakerRuntimeEndpoint}  from "./sagemaker"
import {LambdaHandlers} from "./lambda"


export class IpcAiSaasStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.templateOptions.description = '(SO8016) - IP Camera AI SaaS Service Stack (Face Recognition). Template version v1.0.0';

    /**
     * Deployment Instance Type Selection
     */
    const deployInstanceType = new cdk.CfnParameter(this, 'deployInstanceType', {
        description: 'Please choose your desired deployment instance type',
        type: 'String',
        default: 'ml.g4dn.xlarge',
        allowedValues: [
            'ml.g4dn.xlarge',
        ]
    });

    const faceDetectorModel = new cdk.CfnParameter(this, 'faceDetectorModel', {
        description: 'Please choose your desired face detector model',
        type: 'String',
        default: 'retinaface_mnet025_v2',
        allowedValues: [
            'retinaface_r50_v1',
            'retinaface_mnet025_v2'
        ]
    });

    const faceRepresenterModel = new cdk.CfnParameter(this, 'faceRepresenterModel', {
        description: 'Please choose your desired face representation model',
        type: 'String',
        default: 'MobileFaceNet',
        allowedValues: [
            'MobileFaceNet',
            'LResNet34E-IR',
            'LResNet50E-IR',
            'LResNet100E-IR'
        ]
    });


    const faceConfidenceThreshold = new cdk.CfnParameter(this, 'faceConfidenceThreshold', {
        description: 'Please configure the confidence threshold of faces (only faces above this threshold will be further represented)',
        type: 'String',
        default: '0.70',
        allowedValues: [
            '0.70',
            '0.80',
            '0.90'
        ]
    });


    /**
     * S3 Bucket Provision
     */
    const imageAssets = new s3.Bucket(
        this,
        'imageAssets',
        {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
        }
    );


    /**
     * DynamoDB Provision
     */
    const faces = new dynamodb.Table(
        this,
        'faces',
        {
            partitionKey: {
                name: 'activity_id',
                type: dynamodb.AttributeType.STRING
            },
            sortKey: {
                name: 'image_id',
                type: dynamodb.AttributeType.STRING
            },
            tableName: 'faces',
            billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
        }
    );


    /**
     * Create SageMaker runtime endpoint with auto-scaling enabled.
     */
    const sagemakerStack = new SageMakerRuntimeEndpoint(
        this,
        'sagemakerStack',
        {
            deployInstanceType: deployInstanceType.valueAsString,
            faceDetectorModel: faceDetectorModel.valueAsString,
            faceRepresenterModel: faceRepresenterModel.valueAsString,
            faceConfidenceThreshold: faceConfidenceThreshold.valueAsNumber
        }
    );


    /**
     * Create Lambda Functions
     */
    const lambdaStack = new LambdaHandlers(
        this,
        'lambdaStack',
        {
            imageAssets: imageAssets,
            facesTableName: faces.tableName,
            sagemakerInferenceEndpointName: sagemakerStack.faceRecgnitionEndpointName,
        }
    );


    /**
     * Create API Gateway
     */
    const faceRecognitionAPIRouter = new agw.RestApi(
        this,
        'faceRecognitionAPIRouter',
        {
            endpointConfiguration: {
                types: [agw.EndpointType.REGIONAL]
            },
            defaultCorsPreflightOptions: {
                allowOrigins: agw.Cors.ALL_ORIGINS,
                allowMethods: agw.Cors.ALL_METHODS
            }
        }
    );
    faceRecognitionAPIRouter.root.addResource('upload').addMethod('POST', new agw.LambdaIntegration(lambdaStack.imageUploader));
    faceRecognitionAPIRouter.root.addResource('activity').addMethod('POST', new agw.LambdaIntegration(lambdaStack.activitySummary));
    faceRecognitionAPIRouter.root.addResource('query').addMethod('POST', new agw.LambdaIntegration(lambdaStack.faceQuery));

  }
}
