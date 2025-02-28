#!/bin/bash
set -e 

# Move into the correct directory
cd "$(dirname "$0")"

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "Error: Unable to fetch AWS Account ID. Check AWS credentials."
  exit 1
fi

ECR_REPO_NAME="aiml-dissertation-customer-segmentation-docker"
IMAGE_TAG="latest"
ECR_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG"

echo "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
echo "ECR_IMAGE_URI: $ECR_IMAGE_URI"

# Step 1: Verify AWS Authentication
aws sts get-caller-identity || { echo "AWS authentication failed. Check your credentials."; exit 1; }

# Step 2: Check if ECR Repository Exists
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "ECR repository does not exist. Creating..."
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" || { echo "Failed to create ECR repository"; exit 1; }
else
  echo "ECR repository already exists."
fi

# Step 3: Authenticate Docker with ECR
echo "Logging in to AWS ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Step 4: Ensure Docker is Running
docker info >/dev/null 2>&1 || { echo "Docker is not running! Start Docker and retry."; exit 1; }

# Step 5: Build the Docker Image
echo "Building the Docker image..."
docker build -t "$ECR_REPO_NAME" .

# Step 6: Verify Image Built Successfully
if ! docker images "$ECR_REPO_NAME" | grep latest; then
  echo "Docker image $ECR_REPO_NAME:latest was not built successfully."
  exit 1
fi

# Step 7: Tag the Docker Image
docker tag "$ECR_REPO_NAME:latest" "$ECR_IMAGE_URI"

# Step 8: Push the Docker Image to ECR
echo "Pushing Docker image to ECR..."
docker push "$ECR_IMAGE_URI"

echo "Deployment complete!"
