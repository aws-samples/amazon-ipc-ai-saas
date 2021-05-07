# Sagemaker Neo模型优化指南

### 目录
* [模型准备](#模型准备)
* [Neo编译作业](#Neo编译作业)
* [EC2推理速度测试](#EC2推理速度测试)
    * [启动EC2实例](#启动EC2实例)
    * [推理测试](#推理测试)


### 模型准备
在利用Sagemaker Neo编译模型之前，首先需要根据神经网络的框架准备模型，具体可以参考官方指南
[Compile and Deploy Models with Neo](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-compilation-preparing-model.html), 
不同的框架对模型的要求各不相同，以MXNet框架为例，其模型的要求描述为：
```
MXNet models must be saved as a single symbol file *-symbol.json and a single parameter *.params files.
```

为了说明整个Sagemaker Neo优化的详细流程，该文档以`yolo3_mobilenet1.0_coco`人形检测模型为例进行模型准备，Neo编译，以及测试。

模型准备分为两步，分别如下：
- 第一步： 下载`yolo3_mobilenet1.0_coco`人形检测模型；
```
wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-0000.params
wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-symbol.json
```

- 第二步：将上述两个文件打成`.tar.gz`包，即执行如下python脚本；
```
import tarfile

tar = tarfile.open("body_detector_yolo3_mobilenet1.0_coco.tar.gz", "w:gz")
for name in ["body_detector_yolo3_mobilenet1.0_coco-0000.params", "body_detector_yolo3_mobilenet1.0_coco-symbol.json"]:
    tar.add(name)
tar.close()
```
脚本执行完毕，会在当前目录生成名为`body_detector_yolo3_mobilenet1.0_coco.tar.gz`的文件，该文件为Sagemaker Neo编译任务的输入。



### Neo编译作业

Sagemaker Neo编译作业可以直接在Sagemaker控制台实现，它的输入是一个S3桶路径，优化后的模型导出也是一个S3桶路径，整个过程如下：
1. 将[模型准备](#模型准备)中生成的`body_detector_yolo3_mobilenet1.0_coco.tar.gz`上传至任意一个指定的S3桶路径，如`s3://object-det-neo/input/body_detector_yolo3_mobilenet1.0_coco.tar.gz`;
1. 进入Sagemaker控制台，点击左侧导航栏`推理-编译作业`，创建编译作业，输入作业名称，创建具有S3桶访问权限的IAM角色，输入配置中的选项包括：
    * 模型构件的位置: 模型存储的S3桶路径
    * 数据输入配置: 即模型的推理时输入尺寸，该指南中以宽高比为3:2的图像为基准，输入尺度为`{"data": [1, 3, 416, 624]}`
    * 机器学习框架: 由于该指南以MXNet为例，故选择MXNet
1. 输出配置选项包括：
    * 目标设备/目标平台：选择目标平
    * 操作系统：选择LINUX;
    * 架构：选择X86_64
    * 加速器：选择NVIDIA
    * 编译器选项：输入`{"gpu-code": "sm_75", "trt-ver": "7.0.0", "cuda-ver": "10.1"}`
    * S3输出位置：指定优化后的模型输出，如`s3://object-det-neo/output/`
    * 加密密钥：保持默认（无自定义加密）

最后点击提交，等待3-5分钟，待编译作业状态显示完成后编译后的模型（如`s3://object-det-neo/output/body_detector_yolo3_mobilenet1.0_coco-LINUX_X86_64_NVIDIA.tar.gz`）便会输出到指定的S3桶位置。


### EC2推理速度测试

#### 启动EC2实例
进入EC2控制台，启动实例，选择Amazon系统镜像（AMI）为`Deep Learning AMI (Amazon Linux 2) Version 44.0 - ami-01f1817a8a0c23c2e`，
选择实例类型为`g4dn.xlarge`，点击下一步配置实例详细信息，保持默认，点击下一步添加存储，保持默认根卷大小95GiB，点击下一步添加标签（可选），
点击下一步配置安组，保持默认，点击审核和启动，选择现有密钥对或创建新密钥对，点击启动。

稍等几分钟，等待实例的状态检查初始化完成后，通过terminal链接该实例：
```
ssh -i "your_key.pem" ec2-user@ec2-xxx-xxx-xx-xxx.compute-1.amazonaws.com
```

#### 推理测试
通过SSH登录链接到实例之后，下载推理测试的代码和模型（未经过Neo优化的和经过Neo优化的），具体执行命令如下所示：
```
git clone https://github.com/aws-samples/amazon-ipc-ai-saas.git
cd amazon-ipc-ai-saas/source/neo
mkdir -p models/human_body_detector
cd models/human_body_detector
wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-0000.params
wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-symbol.json

cd ../../
mkdir -p models/human_body_detector_neo
cd models/human_body_detector_neo
wget -c https://ipc-models-zoo.s3.amazonaws.com/body-detector/body_detector_yolo3_mobilenet1.0_coco-LINUX_X86_64_NVIDIA.tar.gz
tar -zxvf body_detector_yolo3_mobilenet1.0_coco-LINUX_X86_64_NVIDIA.tar.gz

```
上述命令执行之后，在`neo/`目录下的结构如下所示：
```
.
├── eval.py
├── models
│   ├── human_body_detector
│   │   ├── body_detector_yolo3_mobilenet1.0_coco-0000.params
│   │   └── body_detector_yolo3_mobilenet1.0_coco-symbol.json
│   └── human_body_detector_neo
│       ├── compiled.meta
│       ├── compiled_model.json
│       ├── compiled.params
│       ├── compiled.so
│       └── manifest
└── test_1280x1920x3.jpg
```

安装`neo-ai-dlr`软件和`gluoncv`依赖包，参考[https://github.com/neo-ai/neo-ai-dlr/releases](https://github.com/neo-ai/neo-ai-dlr/releases)；
这里测试平台为Amazon g4dn.xlarge，安装命令如下：
```
wget -c https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.9.0/gpu/dlr-1.9.0-py3-none-any.whl
source activate mxnet_latest_p37
pip3 install dlr-1.9.0-py3-none-any.whl
pip3 install gluoncv==0.8.0
```

退回到`neo/`目录，执行速度评估脚本```eval.py```，如下所示：
```
python3 eval.py
```

运行结果会直接打印在terminal之上，同时也会将推理的结果绘制出来并保存到当前目录下（未经Sagemaker Neo优化的检测结果```body_det_vis.jpg```，
经Sagemaker Neo优化后的推理模型检测结果```body_det_vis_with_neo.jpg```）。
运行时间开销输出结果如下：
```
[NEO Optimization Disabled] Time Cost per Frame (input size = 1x3x416x624) = 23.388335704803467 ms
[NEO Optimization Enabled] Time Cost per Frame (input size = 1x3x416x624) = 10.05416750907898 ms
```
Sagemaker Neo优化过的模型可以将推理速度提升一倍以上，该推理时间不含将图像进行base64解码以及resize的部分。

> 注意：在测试结束之后，关闭该实例，避免产生不必要的费用。

