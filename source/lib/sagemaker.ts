import {Construct, Stack} from "@aws-cdk/core";
import * as cdk from "@aws-cdk/core";
import * as iam from "@aws-cdk/aws-iam";
import * as sagemaker from '@aws-cdk/aws-sagemaker';
import {CfnEndpoint} from '@aws-cdk/aws-sagemaker';


export interface SageMakerRuntimeEndpointProps {
    readonly deployInstanceType: string,
    readonly faceDetectorModel: string,
    readonly faceRepresenterModel: string,
    readonly faceConfidenceThreshold: string,
}


export class SageMakerRuntimeEndpoint extends Construct {
    public readonly faceRecgnitionEndpointName: string;

    constructor(scope: Construct, id: string, props: SageMakerRuntimeEndpointProps) {
        super(scope, id);

        /**
         * Create SageMaker Runtime Endpoint Execution Role
         */
        const faceRecognitionRole = new iam.Role(
            this,
            'smFaceRecognitionRole',
            {
                assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
                managedPolicies: [
                    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
                    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryFullAccess'),
                    iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess'),
                ]
            }
        );


        /**
         * Create Runtime Inference Model/Model Configuration/Endpoint
         */
        const imageUrl = `503146276818.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com.cn/ipc-ai-saas-face-recognition-gpu:latest`;
        const faceRecognitionInferenceModel = new sagemaker.CfnModel(
            this,
            'faceRecognitionInferenceModel',
            {
                executionRoleArn: faceRecognitionRole.roleArn,
                containers: [
                    {
                        image: imageUrl.toString(),
                        mode: 'SingleModel',
                        environment: {
                            FACE_DETECTOR_MODEL: props.faceDetectorModel,
                            FACE_REPRESENT_MODEL: props.faceRepresenterModel,
                            FACE_CONFIDENCE_THRESHOLD: props.faceConfidenceThreshold,
                        }
                    }
                ],
            }
        );

        const faceRecognitionInferenceEndpointConfig = new sagemaker.CfnEndpointConfig(
            this,
            'faceRecognitionInferenceEndpointConfig',
            {
                productionVariants: [{
                    initialInstanceCount: 1,
                    initialVariantWeight: 1,
                    instanceType: props.deployInstanceType,
                    modelName: faceRecognitionInferenceModel.attrModelName,
                    variantName: 'AllTraffic',
                }]
            }
        );

        const faceRecognitionInferenceEndpoint = new sagemaker.CfnEndpoint(
            this,
            'faceRecognitionInferenceEndpoint',
            {
                endpointConfigName: faceRecognitionInferenceEndpointConfig.attrEndpointConfigName
            }
        );
        this.faceRecgnitionEndpointName = faceRecognitionInferenceEndpoint.attrEndpointName;


    }
}




