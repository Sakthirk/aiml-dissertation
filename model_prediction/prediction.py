import joblib
import boto3
import io
import pandas as pd

# # Load the saved model and preprocessor
# model = joblib.load(r"D:\BITS\Dissertation\MidSem\model_prediction\xgb_model.joblib")
# preprocessor = joblib.load(r"D:\BITS\Dissertation\MidSem\model_prediction\preprocessor.joblib")


session = boto3.Session(
    aws_access_key_id="AKIAU5LH6DUKYDHGBCGV",
    aws_secret_access_key="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",
    region_name="us-east-1"
)


# AWS S3 Credentials (ensure they are configured correctly)
s3_client = session.client('s3')

# S3 bucket and file details
bucket_name = "aiml-dissertation"
xgboost_file_key = "trained_model/xgb_model.joblib"  # Example: "models/xgb_model.joblib"
preprocessor_file_key = "trained_model/preprocessor.joblib"


def download_file(file_key):
    # Download the file into memory
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_stream = response['Body']

    # Load the model using joblib
    return joblib.load(io.BytesIO(file_stream.read()))


xgboost_model = download_file(xgboost_file_key)
preprocessor = download_file(preprocessor_file_key)


# Sample input data (replace with actual values)
data_dict = {
    "student_country": ["USA"],
    "preferred_contact_time": ["Morning"],
    "preferred_university": ["Harvard"],
    "highest_qualification": ["Bachelor's"],
    "destination_country": ["UK"],
    "area_of_study": ["Computer Science"],
    "intake": ["Fall"],
    "budget": [1000],
    "english_test": ["IELTS"],
    "english_test_score": [7.5],
    "time_to_study": [36],
    "intro_source": ["Google"],
    "app_used": ["No"],
    "sponsor": ["Self"],
    "event_attended": ["No"],
    "standardized_test_Score": [100]  # New feature
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame(data_dict)

# Transform the input data using the preprocessor
input_transformed = preprocessor.transform(input_df)

# Get prediction from the model
prediction = xgboost_model.predict(input_transformed)

# Print the result
print("Predicted Target:", prediction[0])
