import * as cdk from '@aws-cdk/core';
import * as lambda from '@aws-cdk/aws-lambda';
import * as agw from '@aws-cdk/aws-apigateway';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as sagemaker from '@aws-cdk/aws-sagemaker';


export class IpcAiSaasStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.templateOptions.description = '(SO8016) - IP Camera AI SaaS Service Stack. Template version v1.0.0';

    // user choose service type
    const applicationType = new cdk.CfnParameter(this, 'applicationType', {
        description: 'Please choose your desired service type',
        type: 'String',
        default: 'face-detection',
        allowedValues: [
            'face-detection',
            'body-detection',
            'face-comparison'
        ]
    });

    // user choose the deploy instance
    // price reference: https://www.amazonaws.cn/sagemaker/pricing/
    const deployInstanceType = new cdk.CfnParameter(this, 'deployInstanceType', {
        description: 'Please specify the instance type for hosting inference service (ml.c5.xlarge: CPU instance, ml.g4dn.xlarge: GPU instance)',
        type: 'String',
        default: 'ml.g4dn.xlarge',
        allowedValues: [
            'ml.m5.xlarge',
            'ml.g4dn.xlarge',
        ]
    });

    // user choose detection model
    const detectorModelName = new cdk.CfnParameter(this, 'detectorModelName', {
        description: 'Model name for object detector (only valid for face-detection/body-detection scenarios)',
        type: 'String',
        default: 'yolo3_darknet53_coco',
        allowedValues: [
            'ssd_512_resnet50_v1_coco',
            'yolo3_darknet53_coco',
            'yolo3_mobilenet1.0_coco',
            'faster_rcnn_fpn_resnet101_v1d_coco'
        ]
    });

    // user choose face comparison model, detector model + representation model
    const faceDetectAndCompareModelName = new cdk.CfnParameter(this, 'faceDetectAndCompareModelName', {
        description: 'Model name for face detection and comparison (only valid for face-comparison scenario)',
        type: 'String',
        default: 'retinaface_mnet025_v2+MobileFaceNet',
        allowedValues: [
            'retinaface_mnet025_v2+LResNet100E-IR',
            // 'retinaface_mnet025_v2+LResNet50E-IR',
            // 'retinaface_mnet025_v2+LResNet34E-IR',
            'retinaface_mnet025_v2+MobileFaceNet',

            // 'retinaface_r50_v1+LResNet100E-IR',
            // 'retinaface_r50_v1+LResNet50E-IR',
            // 'retinaface_r50_v1+LResNet34E-IR',
            'retinaface_r50_v1+MobileFaceNet',
        ]
    });

    // user choose whether to save all request events
    const saveRequestEvents = new cdk.CfnParameter(this, 'saveRequestEvents', {
        description: 'Whether to save all request images and corresponding response for close-loop improvement',
        type: 'String',
        default: 'No',
        allowedValues: [
            'Yes',
            'No',
        ]
    });

    // build the mapping relationship between instance type and environment (cpu/gpu) configuration
    // NINGXIA region only support G series GPU, no P series GPU
    // BEIJING region support both G and P series GPU
    const map = new cdk.CfnMapping(this, 'instanceEnvMapping', {
      mapping: {
        'ml.m5.xlarge': { value: 'cpu' },
        'ml.g4dn.xlarge': { value: 'gpu' },
      },
    });

    /*-------------------------------------------------------------------------------*/
    /*-------------------          S3 Bucket Provision          ---------------------*/
    /*-------------------------------------------------------------------------------*/
    const events = new s3.Bucket(
        this,
        'events',
        {
            bucketName: `ipc-ai-saas-${applicationType.valueAsString}-events`,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
        }
    );

    /*-------------------------------------------------------------------------------*/
    /*---------  Sagemaker Model/Endpoint Configuration/Endpoint Provision  ---------*/
    /*-------------------------------------------------------------------------------*/
    const sagemakerExecuteRole = new iam.Role(
        this,
        'sagemakerExecuteRole',
        {
            roleName: `ipc-ai-saas-${applicationType.valueAsString}-sagemaker-execution-role`,
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryFullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess'),
            ]
        }
    );

    // configure container image name
    new cdk.CfnCondition(this,
        'IsChinaRegionCondition',
        { expression: cdk.Fn.conditionEquals(cdk.Aws.PARTITION, 'aws-cn')});

    const computeEnv = map.findInMap(`${deployInstanceType.valueAsString}`, 'value');


    const imageUrl = cdk.Fn.conditionIf(
        'IsChinaRegionCondition',
        `753680513547.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com.cn/ipc-ai-saas-${applicationType.valueAsString}-${computeEnv}:latest`,
        `366590864501.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com/ipc-ai-saas-${applicationType.valueAsString}-${computeEnv}:latest`
    );

    // create model
    const sagemakerEndpointModel = new sagemaker.CfnModel(
        this,
        'sagemakerEndpointModel',
        {
            modelName: `ipc-ai-saas-${applicationType.valueAsString}-endpoint-model`,
            executionRoleArn: sagemakerExecuteRole.roleArn,
            containers: [
                {
                    image: imageUrl.toString(),
                    mode: 'SingleModel',
                    environment: {
                        FACE_DETECTION_AND_COMPARISON_MODEL_NAME: `${faceDetectAndCompareModelName.valueAsString}`,
                        OBJECT_DETECTION_MODEL_NAME: `${detectorModelName.valueAsString}`,
                    }
                }
            ],
        }
    );

    // create endpoint configuration
    const sagemakerEndpointConfig = new sagemaker.CfnEndpointConfig(
        this,
        'sagemakerEndpointConfig',
        {
            endpointConfigName: `ipc-ai-saas-${applicationType.valueAsString}-endpoint-config`,
            productionVariants: [{
                initialInstanceCount: 1,
                initialVariantWeight: 1,
                instanceType: `${deployInstanceType.valueAsString}`,
                modelName: sagemakerEndpointModel.attrModelName,
                variantName: 'AllTraffic',
            }]
        }
    );

    // create endpoint
    const sagemakerEndpoint = new sagemaker.CfnEndpoint(
        this,
        'sagemakerEndpoint',
        {
            endpointName: `ipc-ai-saas-${applicationType.valueAsString}-endpoint`,
            endpointConfigName: sagemakerEndpointConfig.attrEndpointConfigName
        }
    );


    /*-------------------------------------------------------------------------------*/
    /*---------------    Lambda Function & API Gateway Provision   ------------------*/
    /*-------------------------------------------------------------------------------*/
    const lambdaAccessPolicy = new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
            "sagemaker:InvokeEndpoint",
            "s3:GetObject",
            "s3:PutObject",
        ],
        resources: ["*"]
    });
    lambdaAccessPolicy.addAllResources();

    // lambda function provision
    const ipcSaaSHandler = new lambda.Function(
        this,
        'ipcSaaSHandler',
        {
            functionName: `ipcSaaSHandler-${applicationType.valueAsString}`,
            code: new lambda.AssetCode( 'lambda'),
            handler: 'main.handler',
            runtime: lambda.Runtime.PYTHON_3_8,
            environment: {
                SAGEMAKER_ENDPOINT_NAME: sagemakerEndpoint.attrEndpointName,
                SERVICE_TYPE: `${applicationType.valueAsString}`,
                OBJECT_DETECTOR_MODEL_NAME: `${detectorModelName.valueAsString}`,
                EVENTS_S3_BUCKET_NAME: events.bucketName,
                REQUEST_EVENTS_SNAPSHOT_ENABLED: `${saveRequestEvents.valueAsString}`,
            },
            timeout: cdk.Duration.minutes(10),
            memorySize: 512,
        }
    );
    ipcSaaSHandler.addToRolePolicy(lambdaAccessPolicy);

    // api gateway provision
    const apiRouter = new agw.RestApi(
        this,
        'apiRouter',
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
    apiRouter.root.addResource('inference').addMethod('POST', new agw.LambdaIntegration(ipcSaaSHandler));

  }
}
