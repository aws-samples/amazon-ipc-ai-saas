# 智能IP摄像头SaaS服务解决方案

## 目录
* [方案介绍](#方案介绍)
* [方案架构](#方案架构)
* [方案部署](#方案部署)
  * [部署须知](#部署须知)
  * [部署参数](#部署参数)
  * [基于Cloudformation部署](#基于Cloudformation部署)
* [实现细节](#实现细节)
  * [基于Darknet框架训练yolov4模型](#基于Darknet框架训练yolov4模型)
  * [TensorRT模型准备](#TensorRT模型准备)
* [API调用代码示例](#API调用代码示例)
* [并发测试](#并发测试)
* [安全](#安全)
* [许可](#许可)


## 方案介绍
该解决方案基于Amazon S3，Amazon Lambda，Amazon API Gateway，Amazon SageMaker等组件，
旨在为智能IP监控行业（IP Camera）提供人工智能算法赋能。该解决方案提供了人形检测， 宠物检测（猫/狗检测）
和车辆检测三种算法服务，AI算法是封装在ECS Container镜像中进行加载部署为SaaS（Software as a Service）
形式提供给用户调用，具有易拓展，可插拔的优点。用户基于该解决方案架构可以任意进行扩展，该解决方案端到端
地提供了一整套云端SaaS服务，用户可以通过设置将生产环境中的数据及其推理结果存储至Amazon S3桶，
为后期算法的迭代升级提供数据基石。

该解决方案支持的功能如下：
- [x] 人形检测
- [x] 宠物检测（猫/狗）
- [x] 车辆检测

三种算法均基于Darknet Yolo-v4进行训练，并转化为TensorRT版本加速推理。

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
- 整个部署过程耗时约为 15-20 分钟。

#### 部署参数

在解决方案部署时，需要指定`applicationType`， `saveRequestEvents`参数:

| 参数                     | 默认值                                               | 描述                                                        |
|-------------------------|-----------------------------------------------------|------------------------------------------------------------|
| `applicationType`       | `persons-detection`  | 配置该解决方案的SaaS服务类型，可选值为 `persons-detection`，`pets-detection`和`vehicles-detection`|
| `saveRequestEvents`     | `No`                 | 配置是否将每一次调用推理时的输入和响应存储至S3桶，可选值为 `Yes`, `No` |


#### 基于Cloudformation部署

请参考下述步骤来基于Cloudformation进行部署：

1. 登录AWS管理控制台，切换到您想将该解决方案部署到的区域；

1. 点击下述按钮（中国与海外）来开启部署；

    - **中国区域 (cn-north-1, cn-northwest-1)**

    [![Launch Stack](launch-stack.svg)](https://console.amazonaws.cn/cloudformation/home?region=cn-north-1#/stacks/create/template?stackName=IPCSolutionStack&templateURL=https://aws-gcr-solutions.s3.cn-north-1.amazonaws.com.cn/amazon-ipc-ai-saas/latest/IpcAiSaasStack.template)

    - **标准（Standard）区域 (us-east-1, us-west-2)**

    [![Launch Stack](launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=IPCSolutionStack&templateURL=https://aws-gcr-solutions.s3.amazonaws.com/amazon-ipc-ai-saas/latest/IpcAiSaasStack.template)

1. 点击 **下一步**. 根据您需要可以更改堆栈名称；

1. 点击 **下一步**. 配置堆栈选项 (可选)；

1. 点击 **下一步**. 审核堆栈配置，勾选 **我确认，AWS CloudFormation 可能创建具有自定义名称的 IAM 资源**，点击 **创建堆栈** 开启创建；

**注意: 当您不再需要该解决方案时，您可以直接从Cloudformation控制台删除它。**


## 实现细节
> 主要阐述如何训练Darknet yolov4目标检测模型，及如何将Darknet yolov4模型转化为TensorRT版本，该小节主要是为了
获得目标检测的TensorRT模型，用来将其封装到ECS推理镜像中。

#### 基于Darknet框架训练yolov4模型
1. 在`Darknet`训练完`Yolo-v4`模型之后，我们需要将其转化为通用的`Tensorflow`模型，并进一步将`Tensorflow`模型
   转化为TensorRT版本，这些转化过程以及最后的`TensorRT`推理需要依赖于`tensorflow-gpu`，`libnvinfer-dev=7.1.3-1+cuda11.0`等
   一系列依赖库的安装，下述命令是基于Amazon EC2 g4dn.xlarge实例（Ubuntu 18.04 OS)的环境准备过程：

    ```angular2html
    sudo apt-get update
    sudo apt-get install -y git cmake awscli libopencv-dev python3-pip
    python3 -m pip install --upgrade pip
    
    # install tensorflow-gpu (SHOULD BE VERSION 2.4.0), it matches cuda/cudnn/nvinfer7 versions
    pip3 install tensorflow-gpu==2.4.0
    pip3 install opencv-python==4.5.2.54
    pip3 install easydict==1.9
    
    # add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    sudo apt-get update
    
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt-get update
    
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
    sudo apt install -y ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
    sudo apt-get update
    
    # install development and runtime libraries (~4GB)
    sudo apt-get install -y --no-install-recommends cuda-11-0 libcudnn8=8.0.4.30-1+cuda11.0 libcudnn8-dev=8.0.4.30-1+cuda11.0 --allow-downgrades
    
    # reboot and check GPUs are visible using command: nvidia-smi
    
    # install TensorRT, which requires that libcudnn8 is installed above
    sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 libnvinfer-dev=7.1.3-1+cuda11.0 libnvinfer-plugin7=7.1.3-1+cuda11.0

    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
    ```


#### TensorRT模型准备

#### 1. CUDA_11.1 + CUDNN_8 + TensorRT_7.2.3 推理环境安装

`TensorRT`可以显著加速推理的速度，为了将`Darknet`训练好的`yolov4`模型转化为`TensorRT`版本，我们首先
借助于 [Open Neural Network Exchange](https://github.com/onnx/onnx) ，将`Darknet`模型转化为`onnx`格式，再将
`onnx`格式转化为`TensorRT`格式。

> Reference:
> - https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian
> - https://github.com/jkjung-avt/tensorrt_demos#yolov4

首先进入`Amazon Web Services EC2`控制台，选择`g4dn.xlarge`实例（`Ubuntu 18.04 OS`)并安装软件依赖项（`cuda`, `tensorrt`, `cudnn`等），如下所示：

```angular2html
sudo apt-get update
sudo apt-get install -y git cmake awscli python3-opencv python3-pip
pip3 install --upgrade pip
pip3 install Cython==0.29.24
pip3 install onnx==1.4.1

# Install CUDA 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# PATH and LD_LIBRARY_PATH Configuration
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install pycuda
pip3 install 'pycuda<2021.1'

# Install TensorRT
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/software/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb .
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y tensorrt
sudo apt-get install python3-libnvinfer-dev
```


#### 2. Darknet模型文件转化

首先编译动态库`libyolo_layer.so`，该动态库会在前向推理中被调用。

```angular2html
git clone https://github.com/jkjung-avt/tensorrt_demos.git
cd tensorrt_demos/plugins
make
```

开始模型转化，下载`Darknet`框架中训练好的`yolov4`模型及其对应的配置文件`*.cfg`，
然后将模型转化为`onnx`格式，再转化为`TensorRT`格式。执行命令如下所示：

```angular2html
cd tensorrt_demos/yolo
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/persons_detection/yolov4-persons.cfg .
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/models/persons_detection/yolov4-persons_best.weights .
mv yolov4-persons_best.weights yolov4-persons.weights
python3 yolo_to_onnx.py -m yolov4-persons
python3 onnx_to_tensorrt.py -m yolov4-persons --verbose
```
执行完之后可以生成后缀为`.trt`的模型文件，该文件将会被封装成容器镜像供用户进行AI SaaS部署。



## API调用代码示例



## 并发测试



## 安全
更多信息请参阅 [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications)。

## 许可
该解决方案遵从MIT-0 许可，更多信息请参阅 LICENSE 文件.
