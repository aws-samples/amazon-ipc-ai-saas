
## 定制化深度学习模型及部署指南

### 目录

* [步骤一：部署环境准备](#步骤一：部署环境准备)
* [步骤二：模型镜像修改](#步骤二：模型镜像修改)
* [步骤三：模型编译并Push至ECR仓库](#步骤三：模型编译并Push至ECR仓库)
* [步骤四：修改源码中容器镜像URL](#步骤四：修改源码中容器镜像URL)
* [步骤五：CDK部署](#步骤五：CDK部署)


> **摘要**：该文阐述了如何自定义容器镜像并将其基于Amazon Sagemaker部署为SaaS服务。该解决方案中深度学习模型的推理逻辑全部封装在
> 容器镜像中，故自定义深度学习模型并将其部署主要需要解决两个问题：其一是类似该解决方案中的自带容器镜像构造代码自定义自己
> 的深度学习模型托管代码；其二是将其编译并且上传（Push）至Amazon Elastic Container Registry (ECR)仓库，并在部署
> 的时候将容器镜像的URL指向它即可。下面分步骤介绍整个自定义模型并部署的详细内容。为了保证不同用户不同操作环境的影响，本指
> 南中将依赖于EC2环境进行示范。


### 步骤一：部署环境准备

**1.1 开启EC2实例**

启动新的EC2实例是为了保证不同用户在自定义模型开发过程和部署过程的一致性，用户也可以基于自己的开发环境来进行开发和部署，
而无需开启新的EC2实例。 首先管理员登录`Amazon EC2`控制台，点击左侧导航栏`实例`，选择`启动新实例`，选择系统镜像为
`Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0747bdcabd34c712a `，64位（x86），选择实例类型
`t2.medium`，点击`下一步：配置实例详细信息`，保持默认，点击`下一步：添加存储`，将根卷大小改为128GB，卷类型保持默认，
点击下一步添加标签（可选），点击`下一步：配置安全组`，保持`SSH/TCP/22`端口开放的默认设置，点击`审核和启动`，点击
`启动`，选择已有的密钥对或者创建新的密钥对并**下载保存**（如果是新创建的密钥对，下载后修改其权限
`chmod 400 your_key_name.pem`），点击启动新实例。

**1.2 创建`iam`用户获取`AK`, `SK`**

管理员登录`Amazon IAM`控制台，点击左侧用户，选择`添加用户`，输入用户名，选择访问类型为`编程访问`,`控制台访问`，
点击下一步权限，选择`直接附加现有策略`,勾选上`AmazonS3FullAccess`，`AmazonSageMakerFullAccess`，
`AmazonAPIGatewayAdministrator`，`AWSLambda_FullAccess`，`AmazonElasticContainerRegistryPublicPowerUser`，
点击下一步标签，点击下一步审核，点击创建用户。创建后页面出现一个**下载.csv**的按钮，点击下载并妥善保存好该文件，后面编译和
创建解决方案堆栈会用到该文件中的`AK（Access Key ID)`，`SK（Secret Access Key）`信息。

**1.3 安装`docker`，`git`，`awscli`等依赖包**

SSH登录进EC2，执行以下命令：
```
sudo apt-get update
sudo apt-get install -y git cmake awscli unzip wget

sudo apt-get install -y docker.io
sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker
```
安装完成后执行`docker ps`命令，如果出现
```
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```
则说明docker安装成功。


**1.4 配置`awscli`命令行环境**

配置aws命令行开发环境，执行如下命令：
```
aws configure
```
输入刚才下载的csv文件中的`Access Key ID`和`Secret Access Key`，`default region name`选择解决方案部署的区域名，
如`us-east-1`，`default output format`选择`json`。

### 步骤二：模型镜像修改

首先clone该解决方案的repo：
```
git clone https://github.com/aws-samples/amazon-ipc-ai-saas.git
```

模型镜像修改主要修改的内容包括两方面，一是`Dockerfile`中的安装项和模型的参数下载路径，二是自定义模型的推理逻辑代码。

**2.1 `Dockerfile`修改**

首先在路径`amazon-ipc-ai-saas/source/containers`下创建一个新的文件夹，如名为`customized-application`，其中代码结构与
内置的其他三个容器镜像一致：
```
cd amazon-ipc-ai-saas/source/containers
cp -rf body-detection/ customized-application
```
`Dockerfile`的修改主要为`amazon-ipc-ai-saas/source/containers/customized-application/Dockerfile`文件中的
系统安装项（`apt-get`），pip安装项（`pip3 install`）以及将自定义的神经网络模型下载到路径`/opt/ml/model`中。


**2.2 推理逻辑代码修改**

逻辑代码修改主要修改文件`amazon-ipc-ai-saas/source/containers/customized-application/detector/predictory.py`，
将其中的逻辑推理部分改用自定义模型的推理逻辑。

### 步骤三：模型编译并Push至ECR仓库

**3.1 更改`build-customized-ecr.sh`脚本**

更改镜像的名称，即`build-customized-ecr.sh`脚本中`your-deep-learning-model-name-cpu`
和`your-deep-learning-model-name-gpu`，将其改为自己定义的模型名称，后缀仍然以`-cpu/-gpu`结尾，表示
支持CPU/GPU两种环境下的部署。 然后将容器镜像编译上传至您账号中的ECR仓库。

```
cd deployment
./build-customized-ecr.sh <region_name> <your_deploy_account> latest
```
其中`region_name`，`your_deploy_account`分别表示您部署的区域名和部署账号。

### 步骤四：修改源码中容器镜像URL

由于自定义的模型名称，模型容器镜像所在的ECR仓库路径相比于该解决方案自带的内置模型均有改变，故需在`source/lib/ipc-ai-saas-stack.ts`资源配置堆栈中作出相应的改变，即改变容器镜像的URL，如下所示：

```
const imageUrl = cdk.Fn.conditionIf(
    'IsChinaRegionCondition',
    `<your_account_id_cn>.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com.cn/your-deep-learning-model-name-${computeEnv}:latest`,
    `<your_account_id>.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com/your-deep-learning-model-name-${computeEnv}:latest`
);
```

### 步骤五：CDK部署

在容器镜像成功被Push至ECR仓库中后，同时`source/lib/ipc-ai-saas-stack.ts`资源配置堆栈中部署容器镜
像也作出正确修改后，便可以基于CDK进行解决方案部署，首先安装`npm`和`aws-cdk`：
```
sudo apt-get install -y npm
sudo npm install -g n
sudo n stable
export PATH="$PATH:/usr/local/bin/node"
sudo npm i -g aws-cdk@1.102.0 
```

安装完成后开始部署解决方案：
```
cd source
rm package-lock.json && npm install 
cdk deploy \
--parameters applicationType=body-detection \
--parameters deployInstanceType=ml.g4dn.xlarge \
--parameters saveRequestEvents=No
```

