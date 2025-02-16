FROM public.ecr.aws/lambda/python:3.8

# Install required packages
RUN pip install --no-cache-dir scikit-learn boto3 joblib xgboost pandas numpy

# Copy the training script (assumed to be downloaded from S3 at runtime)
COPY train.py /var/task/train.py

# Set the Lambda runtime entrypoint
CMD ["train.lambda_handler"]

# Allow running locally
ENTRYPOINT ["python", "/var/task/train.py"]