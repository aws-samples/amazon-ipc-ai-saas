
## 定制化深度学习模型及部署指南

### 目录

* [部署环境准备](#部署环境准备)
* [模型镜像修改](#模型镜像修改)
* [模型编译并Push至ECR仓库](#模型编译并Push至ECR仓库)
* [修改源码中容器镜像URL](#修改源码中容器镜像URL)
* [CDK部署](#CDK部署)


> 该文阐述了如何自定义容器镜像并将其基于Amazon Sagemaker部署为SaaS服务。该解决方案中深度学习模型的推理逻辑全部封装在的容器镜像中，故自定义深度学习模型并将其部署主要需要解决两个问题：其一是模拟该解决方案中的自带的容器镜像构造代码自定义自己的深度学习模型托管代码；其二是将其编译并且上传（Push）至Amazon Elastic Container Registry (ECR)仓库，并在部署的时候将容器镜像的URL指向它即可。下面分步骤介绍整个自定义模型并部署的详细内容。为了保证不同用户不同操作环境的影响，本指南中将依赖于EC2环境进行示范。同时该repo源码中已经为您创建了`source/containers/customized-application`目录以及`deployment/build-customized-ecr.sh`脚本文件，分别用来存储您的自定义推理模型逻辑和镜像编译上传的逻辑。

### 部署环境准备

*开启EC2实例*

> 启动新的EC2实例是为了保证不同用户在自定义模型开发过程和部署过程的一致性，用户也可以基于自己的开发环境来进行开发和部署，而无需开启新的EC2实例。

管理员登录Amazon EC2控制台，点击左侧导航栏`实例`，选择`启动新实例`，选择系统镜像为`Amazon Linux 2 AMI (HVM), SSD Volume Type - ami-0d5eff06f840b45e9`，64位（x86），选择实例类型`t2.medium`，点击`下一步：配置实例详细信息`，保持默认，点击`下一步：添加存储`，将根卷大小改为32GB，卷类型保持默认，点击下一步添加标签（可选），点击`下一步：配置安全组`，保持SSH/TCP/22端口开放的默认设置，点击`审核和启动`，点击`启动`，选择已有的密钥对或者创建新的密钥对并下载保存，启动新实例。

*创建`iam`用户获取`AK`, `SK`*

管理员登录Amazon IAM控制台，点击左侧用户，选择`添加用户`，输入用户名，选择访问类型为`编程访问`,`控制台访问`，点击下一步权限，选择`直接附加现有策略`,勾选上`AmazonS3FullAccess`，`AmazonSageMakerFullAccess`，`AmazonAPIGatewayAdministrator`，`AWSLambda_FullAccess`，`AmazonElasticContainerRegistryPublicPowerUser`，点击下一步标签，点击下一步审核，点击创建用户。创建后页面出现一个**下载.csv**的按钮，点击下载并妥善保存好该文件，后面编译和创建解决方案堆栈会用到该文件中的AK（Access Key ID)，SK（Secret Access Key）信息。

*安装`docker`，`git`等依赖包*

SSH登录进EC2，执行以下命令：
```
sudo amazon-linux-extras install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```
安装完成后重启一下机器`sudo reboot`，再次登录进来，执行`docker ps`命令，如果出现
```
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```
则说明docker安装成功。

紧接着安装`git`依赖：
```
sudo yum install -y git
```

*配置`awscli`命令行环境*

配置aws命令行开发环境，执行如下命令：
```
aws configure
```
输入刚才下载的csv文件中的`Access Key ID`和`Secret Access Key`，`default region name`选择解决方案部署的区域名，如`us-east-1`，`default output format`选择`json`。

### 模型镜像修改

首先clone该解决方案的repo：
```
git clone https://github.com/aws-samples/amazon-ipc-ai-saas.git
```

模型镜像修改主要修改的内容包括两方面，一是`Dockerfile`中的安装项和模型的参数下载路径，二是自定义模型的推理逻辑代码。

*`Dockerfile`修改*

`Dockerfile`的修改主要为`source/containers/customized-application/Dockerfile`文件中的系统安装项（`apt-get`），pip安装项（`pip3 install`）以及将自定义的神经网络模型下载到路径`/opt/ml/model`中。

*推理逻辑代码修改*

逻辑代码修改主要修改文件`source/containers/customized-application/detector/predictory.py`，将其中的逻辑推理部分改用自定义模型的推理逻辑。

### 模型编译并Push至ECR仓库

*更改`build-customized-ecr.sh`脚本*

更改镜像的名称，即`build-customized-ecr.sh`脚本中`your_deep_learning_model_name_cpu`和`your_deep_learning_model_name_gpu`，然后将容器镜像编译上传至您账号中的ECR仓库。

```
cd deployment
./build-customized-ecr.sh <region_name> <your_deploy_account> latest
```
其中`region_name`，`your_deploy_account`分别表示您部署的区域名和部署账号。

### 修改源码中容器镜像URL

由于自定义的模型名称，模型容器镜像所在的ECR仓库路径相比于该解决方案自带的内置模型均有改变，故需在`source/lib/ipc-ai-saas-stack.ts`资源配置堆栈中作出相应的改变，即改变容器镜像的URL，如下所示：

```
const imageUrl = cdk.Fn.conditionIf(
    'IsChinaRegionCondition',
    `<your_account_id_cn>.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com.cn/your_deep_learning_model_name:latest`,
    `<your_account_id>.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com/your_deep_learning_model_name:latest`
);
```

### CDK部署

在容器镜像成功被Push至ECR仓库中后，同时`source/lib/ipc-ai-saas-stack.ts`资源配置堆栈中部署容器镜像也作出正确修改后，便可以基于CDK进行解决方案部署，首先安装`npm`和`aws-cdk`：
```
sudo amazon-linux-extras install -y epel 
sudo yum install -y npm nodejs
npm install -g aws-cdk@1.102.0 
```

安装完成后开始部署解决方案：
```
cd source
rm package-lock.json && npm install 
cdk deploy \
--parameters applicationType=body-detection \
--parameters deployInstanceType=ml.g4dn.xlarge \
--parameters detectorModelName=<your_defined_model_name> \
--parameters faceDetectAndCompareModelName=retinaface_mnet025_v2+MobileFaceNet \
--parameters saveRequestEvents=No
```

