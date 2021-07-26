## 基于Darknet Yolo-v4训练目标检测模型以及利用TensorRT加速推理

### 内容
* [训练EC2装机指南](#训练EC2装机指南)
* [TensorRT模型转换](#TensorRT模型转换)
* [TensorRT推理](#TensorRT推理)


> 官方参考文档：
> - https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10
> - https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#benefits

### 训练EC2装机指南
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


### 利用TensorRT加速推理

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