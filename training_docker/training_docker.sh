#!/bin/bash
set -e 

# Move into the correct directory
cd "$(dirname "$0")"


# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
ECR_REPO_NAME="aiml-dissertation-customer-segmentation-model-training-docker"
IMAGE_TAG="latest"
ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/LambdaExecutionRole"


echo "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
echo "ECR_IMAGE_URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG"



# Step 1: Create ECR Repository (If Not Exists)
echo "Checking if ECR repository exists..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION >/dev/null 2>&1

if [ $? -ne 0 ]; then
  echo "ECR repository does not exist. Creating..."
  aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION
else
  echo "ECR repository already exists."
fi

# Step 2: Authenticate Docker with ECR
echo "Logging in to AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 3: Build the Docker Image
echo "Building the Docker image..."
docker build -t $ECR_REPO_NAME .

# Step 4: Tag the Docker Image
ECR_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG"
docker tag $ECR_REPO_NAME:latest $ECR_IMAGE_URI

# Step 5: Push the Docker Image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_IMAGE_URI

echo "Deployment complete!"
