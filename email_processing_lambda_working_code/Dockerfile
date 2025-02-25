FROM public.ecr.aws/lambda/python:3.9


RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    python3-devel \
    libffi-devel \
    make \
    && yum clean all


# Install required packages
RUN pip install --no-cache-dir scikit-learn==1.6.1 boto3 joblib xgboost pandas numpy fuzzywuzzy

RUN pip install --no-cache-dir spacy && \
    python -m spacy download en_core_web_sm

# Copy the training script (assumed to be downloaded from S3 at runtime)
COPY email_processing.py /var/task/email_processing.py

# Set the Lambda runtime entrypoint
CMD ["email_processing.lambda_handler"]

# Allow running locally
# ENTRYPOINT ["python", "/var/task/email_processing.py"]