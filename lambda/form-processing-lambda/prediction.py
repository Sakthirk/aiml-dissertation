import joblib
import boto3
import io
import pandas as pd
import json
import xgboost
from datetime import datetime
from decimal import Decimal
import numpy as np
import uuid
import pytz
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)


s3_client = boto3.client('s3')

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  # Change to your region
Predictions_table = dynamodb.Table("Predictions")


def download_model_files(bucket_name, file_key):
    # Download the file into memory
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_stream = response['Body']
    
    logging.info(f"Downloading {file_key} from {bucket_name}...")

    try:
        # Read and load the model
        model = joblib.load(io.BytesIO(file_stream.read()))
        logging.info(f"Successfully loaded {file_key}.")
        
        return model
    except Exception as e:
        logging.error(f"Error loading {file_key}: {e}")
        return None
    
def download_models():

    # S3 bucket and file details
    bucket_name = "aiml-dissertation"
    xgboost_file_key = "trained_model/xgb_model.joblib"
    preprocessor_file_key = "trained_model/preprocessor.joblib"

    xgboost_model = download_model_files(bucket_name,xgboost_file_key)
    preprocessor = download_model_files(bucket_name,preprocessor_file_key)

    return xgboost_model, preprocessor

def process_budget(budget_str):
    if pd.isna(budget_str):
        return np.nan

    match = re.match(r'(\d+(?:,\d{3})*)\s*-\s*(\d+(?:,\d{3})*)\s*USD', budget_str)
    if match:
        # Extract the lower and upper bounds and convert to integers (remove commas)
        lower_bound = int(match.group(1).replace(",", ""))
        upper_bound = int(match.group(2).replace(",", ""))
        return (lower_bound + upper_bound) / 2  # Return the average of the two bounds

    match_plus = re.match(r'(\d+(?:,\d{3})*)\+?\s*USD', budget_str)
    if match_plus:
        lower_bound = int(match_plus.group(1).replace(",", ""))
        return lower_bound

    return np.nan

def get_intake_with_year(date_str):
    if date_str is None:
        return None
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month, year = date_obj.month, date_obj.year

    # Define intake periods
    if 1 <= month <= 4:
        intake = "Spring"
    elif 5 <= month <= 8:
        intake = "Summer"
    else:
        intake = "Fall"

    return f"{intake} {year}"

def months_between_dates(target_date_str, date_format="%Y-%m-%d"):

    if target_date_str is None:
        return None

    # Get the current date
    current_date = datetime.today()

    # Convert the target date string to a datetime object
    target_date = datetime.strptime(target_date_str, date_format)

    # Calculate the difference in months
    months = (target_date.year - current_date.year) * 12 + (target_date.month - current_date.month)
    
    return months


def convert_dynamodb_compatible(obj):
    """ Recursively convert unsupported types to DynamoDB-compatible types """
    if isinstance(obj, dict):
        return {k: convert_dynamodb_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dynamodb_compatible(i) for i in obj]
    elif isinstance(obj, float):  
        return Decimal(str(obj))  # Convert float to Decimal
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):  
        return int(obj)  # Convert NumPy int to Python int
    elif isinstance(obj, (np.float32, np.float64)):  
        return float(obj)  # Convert NumPy float to Python float
    elif obj is None:
        return None  # Explicitly handle None values
    return obj  # Return other types unchanged


def store_prediction(source_data,features, email, prediction,source):

    features = convert_dynamodb_compatible(features)  # Convert features for DynamoDB
    prediction = convert_dynamodb_compatible(prediction)  # Ensure prediction is converted


    """ Store input features and model prediction in DynamoDB """
    
    # Generate unique ID for each prediction
    prediction_id = str(uuid.uuid4())
    
    # Get IST timezone
    ist = pytz.timezone("Asia/Kolkata")

    # Get current time in UTC
    utc_now = datetime.utcnow()

    # Convert UTC to IST
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist)

    # Format IST time in ISO 8601 format
    timestamp_ist = ist_now.isoformat()

    
    item = {
        "id": prediction_id,
        "email": email,  # New email field,
        "source":source,
        "source_data":source_data,
        "features": features,  # Storing all features in a Map
        "prediction": prediction,
        "timestamp": timestamp_ist  # Store timestamp

    }
    # Insert into DynamoDB
    Predictions_table.put_item(Item=item)
    logging.info(f"Stored Prediction ID: {prediction_id}")

def standardize_score(english_test,english_test_score:float):
    if english_test and english_test_score:
        if english_test == "IELTS":
            # IELTS: Original scale 0-9
            return 1 + (english_test_score - 0) * (10 - 1) / (9 - 0)
        elif english_test == "TOEFL":
            # TOEFL: Original scale 0-120
            return 1 + (english_test_score - 0) * (10 - 1) / (120 - 0)
        else:
            return None  # Handle unexpected test types
    else:
        return None
    
def match_features_to_model_columns(features):

    proficiency_test_score = float(features.get('proficiency_test_score').strip()) if features.get('proficiency_test_score') else None

    feature_dict = {
        "student_country": [features.get('country_of_residence')],
        "preferred_contact_time": [features.get('preferred_contact_time')],
        "preferred_university": [features.get('preferred_university')],
        "highest_qualification": [features.get('highest_qualification')],
        "destination_country": [features.get('desired_country')],
        "area_of_study": [features.get('field_of_study')],
        "intake": [get_intake_with_year(features.get('start_date'))],
        "budget": [process_budget(features.get('budget'))],
        "english_test": [features.get('proficiency_test')],
        "english_test_score": [features.get('proficiency_test_score')],
        "time_to_study": [months_between_dates(features.get('start_date'))],
        "intro_source": [features.get('how_you_know')],
        "app_used": ["No"],
        "sponsor": [features.get('sponsor')],
        "event_attended": ["No"],
        "standardized_test_Score": [standardize_score(features.get('proficiency_test'),proficiency_test_score)]
    }

    

    return feature_dict

def lambda_handler(event, context):

    try:

        for record in event['Records']:
            message = json.loads(json.loads(record['body'])['Message'])
            
            logging.info(f'Received message: {message}')

            source_data = message.get('data')

            logging.info("Downloading models")

            xgboost_model,preprocessor =  download_models()

            # print("Matching features")
            logging.info("Matching features")

            features = match_features_to_model_columns(source_data)
            logging.info(f"features - {json.dumps(features)}")
            
            # Convert dictionary to DataFrame
            input_df = pd.DataFrame(features)

            # Transform the input data using the preprocessor
            input_transformed = preprocessor.transform(input_df)

            logging.info("Prediction started")
            # Get prediction from the model
            # prediction = xgboost_model.predict(input_transformed)

            # Predict probability
            probabilities = xgboost_model.predict_proba(input_transformed)[:, 1]  # Probability of class 1

            prob = probabilities[:10]

            prediction = "Low-Priority" if prob < 0.30 else "Priority" if prob <= 0.70 else "High-Priority"

            logging.info(f"Predicted Target: {prediction}")

            logging.info("Storing prediction")

            store_prediction(source_data,features,source_data.get('email'),prediction,'Form Submission')
            
            logging.info("Prediction stored successfully")            

    except Exception as error:
        print(error)
