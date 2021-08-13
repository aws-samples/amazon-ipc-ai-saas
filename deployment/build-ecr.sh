#!/bin/bash

# Check to see if the required parameters have been provided:
if [ -z "$1" ] || [ -z "$2" ] ||  [ -z "$3" ]; then
    echo "Please provide the region_name, account_id and cpu/gpu version to build the ecr image."
    echo "For example: ./build-ecr.sh <region_name> <account_id> latest"
    exit 1
fi

# Get reference for all important folders
TEMPLATE_DIR="$PWD"
SOURCE_DIR="${TEMPLATE_DIR}/../source"

echo "------------------------------------------------------------------------------"
echo "[Init] Get Env"
echo "------------------------------------------------------------------------------"
REGION=$1
ACCOUNT_ID=$2
IMAGE_TAG=$3

if [[ $1 == cn-* ]];
then
  DOMAIN=$2.dkr.ecr.$1.amazonaws.com.cn
  REGISTRY_ID="727897471807"
  REGISTRY_DOMAIN="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com.cn"

  # images uri
  REGISTRY_URI_CPU="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com.cn/mxnet-inference:1.8.0-cpu-py37-ubuntu16.04"
  REGISTRY_URI_GPU="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com.cn/mxnet-inference:1.8.0-gpu-py37-cu110-ubuntu16.04"
else
  DOMAIN=$2.dkr.ecr.$1.amazonaws.com
  REGISTRY_ID="763104351884"
  REGISTRY_DOMAIN="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com"

  # images uri
  REGISTRY_URI_CPU="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/mxnet-inference:1.8.0-cpu-py37-ubuntu16.04"
  REGISTRY_URI_GPU="${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/mxnet-inference:1.8.0-gpu-py37-cu110-ubuntu16.04"
fi

echo ECR_DOMAIN ${DOMAIN}
echo REGISTRY_URI_CPU ${REGISTRY_URI_CPU}
echo REGISTRY_URI_GPU ${REGISTRY_URI_GPU}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${DOMAIN}
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REGISTRY_DOMAIN}

#############################################################################################
###                     Face Comparison Image Build & Push                                ###
#############################################################################################
echo "------------------------------------------------------------------------------"
echo "[Build] Build Face Recognition Image (GPU Version)                            "
echo "------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
IMAGE_NAME=ipc-ai-saas-face-recognition-gpu
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f containers/faces/Dockerfile containers/faces/ --build-arg REGISTRY_URI=${REGISTRY_URI_GPU}
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOMAIN}/${IMAGE_NAME}:${IMAGE_TAG}

echo "------------------------------------------------------------------------------"
echo "[Push] Push Face Recognition Image (GPU Version)                              "
echo "------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION} >/dev/null 2>&1
docker push ${DOMAIN}/${IMAGE_NAME}:${IMAGE_TAG}

