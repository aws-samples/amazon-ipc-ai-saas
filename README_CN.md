# 智能IP摄像头SaaS服务解决方案

## 目录
* [方案介绍](#方案介绍)
* [方案架构](#方案架构)
* [方案部署](#方案部署)
  * [部署须知](#部署须知)
  * [部署参数](#部署参数)
  * [基于Cloudformation部署](#基于Cloudformation部署)
  * [基于CDK部署](#基于CDK部署)
* [安全](#安全)
* [许可](#许可)

## 方案介绍
该解决方案基于Amazon S3，Amazon Lambda，Amazon API Gateway，Amazon SageMaker等组件，
旨在为智能IP监控行业（IP Camera）提供人工智能算法赋能。该解决方案提供了人脸检测，人形检测，
人脸比较三种算法服务，AI算法是封装在ECS Container镜像中进行加载部署为SaaS（Software as a Service）
形式提供给用户调用，具有易拓展，可插拔的优点。用户基于该解决方案架构可以任意进行扩展，该解决方案端到端
地提供了一整套云端SaaS服务，用户可以通过设置将生产环境中的数据及其推理结果存储至Amazon S3桶，
为后期算法的迭代升级提供数据基石。

该解决方案支持的功能如下：
- [x] 人形检测
- [x] 宠物检测（猫/狗）
- [x] 车辆检测


## 方案架构
![IPC_AI_SaaS_Architecture](architecture.png)

架构图中各个组件的功能描述如下:
1. Amazon API Gateway: 路由用户的请求，用户的请求中携带图片的base64编码；
1. Amazon Lambda Function: 将用户的请求转发到Sagemaker Endpoint进行推理，同时负责将请求的图片和Sagemaker Endpoint推理的结果写入S3桶；
1. Amazon S3 Bucket: 用来存储用户API调用的请求图片和推理响应结果（json格式）；
1. Sagemaker Endpoint + Load Balancing + ML Instances: 负责处理用户的推理请求，基于Sagemaker Endpoint对请求图像实现人脸检测，人形检测和人脸比较，用户可以根据Sagemaker Endpoint的托管机器的工作负载对其进行Auto-Scaling，完成大批量高并发的请求服务。


## 方案部署

#### 部署须知

- 该解决方案在部署过程中会自动地在您的账户中配置Amazon S3 Bucket，API Gateway， Lambda，Sagemaker Model/Configuration/Endpoint等等。
- 整个部署过程耗时约为 10-20 分钟。

#### 部署参数

在解决方案部署时，需要指定`applicationType`，`deployInstanceType`，`detectorModelName`，`faceDetectAndCompareModelName`，`saveRequestEvents`参数:

| 参数                 | 默认值                                             | 描述                                                                                     |
|---------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `applicationType`       | `face-detection`  | 配置该解决方案的SaaS服务类型，可选值为 `face-detection`，`body-detection`， `face-comparison`|
| `deployInstanceType`     | `ml.g4dn.xlarge`  | AI推理服务部署的机型，可选值包括  `ml.m5.xlarge`, `ml.g4dn.xlarge` |
| `detectorModelName`     | `yolo3_darknet53_coco`  | 配置人脸检测和人形检测服务中所使用的检测算法名称，只有`applicationType`值为`face-detection`，`body-detection`时有效，可选值为 `ssd_512_resnet50_v1_coco`, `yolo3_darknet53_coco`, `yolo3_mobilenet1.0_coco`, `faster_rcnn_fpn_resnet101_v1d_coco` |
| `faceDetectAndCompareModelName`     | `retinaface_mnet025_v2+MobileFaceNet`  | 配置人脸比较算法中检测模型和人脸表征模型，仅在`applicationType`值为`face-comparison`时有效，可选值为 `retinaface_mnet025_v2+LResNet100E-IR`, `retinaface_mnet025_v2+MobileFaceNet`,  `retinaface_r50_v1+MobileFaceNet` |
| `saveRequestEvents`     | `No`  | 配置是否将每一次调用推理时的输入和响应存储至S3桶，可选值为 `Yes`, `No` |


#### 基于Cloudformation部署

请参考下述步骤来基于Cloudformation进行部署：

1. 登录AWS管理控制台，切换到您想将该解决方案部署到的区域；

1. 点击下述按钮（中国与海外）来开启部署；

    - 中国区域 (cn-north-1, cn-northwest-1)

    [![Launch Stack](launch-stack.svg)](https://console.amazonaws.cn/cloudformation/home?region=cn-north-1#/stacks/create/template?stackName=IPCSolutionStack&templateURL=https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/amazon-ipc-ai-saas/latest/IpcAiSaasStack.template)

    - 标准（Standard)区域 (us-east-1, us-west-2)

    [![Launch Stack](launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=IPCSolutionStack&templateURL=https://aws-gcr-solutions.s3.amazonaws.com/amazon-ipc-ai-saas/latest/IpcAiSaasStack.template)

1. 点击 **下一步**. 根据您需要可以更改堆栈名称；

1. 点击 **下一步**. 配置堆栈选项 (可选)；

1. 点击 **下一步**. 审核堆栈配置，勾选 **我确认，AWS CloudFormation 可能创建具有自定义名称的 IAM 资源**，点击 **创建堆栈** 开启创建；

> 注意: 当您不再需要该解决方案时，您可以直接从Cloudformation控制台删除它。


#### 基于CDK部署

如果您想基于AWS CDK部署该解决方案，请您确认您的部署环境满足下述前提条件：

* [AWS Command Line Interface](https://aws.amazon.com/cli/)
* Node.js 12.x or 更高版本

在 **source** 文件夹下, 执行下述命令将TypeScript编译成JavaScript；

```
cd source
npm install -g aws-cdk
npm install && npm run build
```

然后您可以执行 `cdk deploy` 命令开启部署该解决方案，如下所示：

```
cdk deploy \
--parameters applicationType=face-detection \
--parameters deployInstanceType=ml.g4dn.xlarge \
--parameters detectorModelName=yolo3_darknet53_coco \
--parameters faceDetectAndCompareModelName=retinaface_mnet025_v2+MobileFaceNet \
--parameters saveRequestEvents=Yes
```

> 注意: 当您不再需要该解决方案时，您可以执行 `cdk destroy` 命令，该命令会将部署账户中该解决方案创建的资源移除掉。


## 安全
更多信息请参阅 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

## 许可
该解决方案遵从MIT-0 许可，更多信息请参阅 LICENSE 文件.

