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
else
  DOMAIN=$2.dkr.ecr.$1.amazonaws.com
fi

echo ECR_DOMAIN ${DOMAIN}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${DOMAIN}


#############################################################################################
###                        Vehicles Detection Image Build & Push                          ###
#############################################################################################
echo "------------------------------------------------------------------------------"
echo "[Build] Build Face Recognition Image (GPU Version)                            "
echo "------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
IMAGE_NAME=ipc-ai-saas-face-recognition-gpu
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f containers/faces/Dockerfile containers/faces/
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOMAIN}/${IMAGE_NAME}:${IMAGE_TAG}

echo "------------------------------------------------------------------------------"
echo "[Push] Push Face Recognition Image (GPU Version)                              "
echo "------------------------------------------------------------------------------"
cd ${SOURCE_DIR}
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION} >/dev/null 2>&1
docker push ${DOMAIN}/${IMAGE_NAME}:${IMAGE_TAG}
