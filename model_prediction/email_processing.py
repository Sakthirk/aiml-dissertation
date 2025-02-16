import json
import boto3
import spacy
from datetime import datetime
import joblib
import io
from fuzzywuzzy import process
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from decimal import Decimal
import pandas as pd
import uuid
import numpy as np
import pytz

session = boto3.Session(
    aws_access_key_id="AKIAU5LH6DUKYDHGBCGV",
    aws_secret_access_key="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",
    region_name="us-east-1"
)


s3_client = session.client('s3')

dynamodb = session.resource("dynamodb", region_name="us-east-1")  # Change to your region
Predictions_table = dynamodb.Table("Predictions")

nlp = spacy.load("en_core_web_sm")

def download_model_files(bucket_name,file_key):
    # Download the file into memory
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_stream = response['Body']

    # Load the model using joblib
    return joblib.load(io.BytesIO(file_stream.read()))

def download_models():

    # S3 bucket and file details
    bucket_name = "aiml-dissertation"
    xgboost_file_key = "trained_model/xgb_model.joblib"
    preprocessor_file_key = "trained_model/preprocessor.joblib"

    xgboost_model = download_model_files(bucket_name,xgboost_file_key)
    preprocessor = download_model_files(bucket_name,preprocessor_file_key)

    return xgboost_model, preprocessor

def download_email_intent_classification_models():
    
    # S3 bucket and file details
    bucket_name = "aiml-dissertation"
    model_file_key = "intent_classification/email_intent_model.joblib"
    vectorizer_file_key = "intent_classification/vectorizer.joblib"

    model = download_model_files(bucket_name,model_file_key)
    vectorizer = download_model_files(bucket_name,vectorizer_file_key)

    return model, vectorizer

def predict_intent(email):
    print("Intent classification")

    model, vectorizer = download_email_intent_classification_models()
    email_transformed = vectorizer.transform([email])
    intent = model.predict(email_transformed)

    return intent[0]


# Predefined lists
COUNTRIES = [
    "Australia", "Canada", "United States", "Germany", "United Kingdom", "India", "New Zealand",
    "France", "Singapore", "Ireland", "Netherlands"
]

QUALIFICATIONS = [
    "bachelor's degree", "master's degree", "high school diploma", "PhD", "associate degree",
    "diploma", "certificate", "advanced diploma", "graduate diploma", "undergraduate degree", "high school diploma"
]

FIELDS = [
    "Computer Science", "Data Science", "Data Analytics", "Machine Learning", "Artificial Intelligence",
    "Software Engineering", "Cybersecurity", "Finance", "Business Administration", "Marketing", "Economics",
    "Accounting", "Human Resources Management", "Psychology", "Sociology", "Political Science", "International Relations",
    "Biology", "Biotechnology", "Environmental Science", "Medicine", "Pharmacy", "Nursing", "Physics", "Chemistry",
    "Mathematics", "Statistics", "Aerospace Engineering", "Mechanical Engineering", "Civil Engineering",
    "Electrical Engineering", "Architecture", "Renewable Energy", "Journalism", "Law", "Education", "Philosophy", "History"
]

# Create PhraseMatcher object
matcher = PhraseMatcher(nlp.vocab)

# Add patterns for qualification matching
patterns = [nlp.make_doc(qualification) for qualification in QUALIFICATIONS]
matcher.add("QUALIFICATIONS", None, *patterns)

def get_field_of_study(email_text):
    """Extract field of study based on key terms or cosine similarity."""
    study_field_sentences = re.findall(
        r"(?:pursue|study|join|interested in|wish to|planning to study|specializing in)\s.*?\.",
        email_text, re.IGNORECASE
    )
    relevant_text = " ".join(study_field_sentences) if study_field_sentences else email_text
    relevant_text = re.sub(r"\bfinance\b", "", relevant_text, flags=re.IGNORECASE)

    exact_matches = [field for field in FIELDS if field.lower() in relevant_text.lower()]
    if exact_matches:
        return exact_matches[0]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([relevant_text] + FIELDS)
    cos_sim = cosine_similarity(vectors[0:1], vectors[1:])
    best_match_index = cos_sim.argsort()[0, -1]
    return FIELDS[best_match_index]

def extract_tests_scores(email_text, doc):
    """Extract proficiency test scores (TOEFL, IELTS, etc.)"""
    english_tests = ["TOEFL", "IELTS", "PTE", "GMAT", "SAT", "ACT"]
    proficiency_test, test_score = None, None

    for ent in doc.ents:
        for test in english_tests:
            if ent.label_ == "CARDINAL" and test.upper() in email_text.upper():
                proficiency_test = test
                test_score = ent.text
                break

    return proficiency_test, test_score

def extract_highest_qualification(email_text, doc):
    """Extract highest qualification using PhraseMatcher."""
    matches = matcher(doc)
    if matches:
        for _, start, end in matches:
            span = doc[start:end]
            return span.text
    return None

def extract_countries(email_text, doc):
    """Extract countries of residence and desired study countries."""
    countries = {"Country of Residence": None, "Desired Country for Studies": None}
    country_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    residence_patterns = [
        r"(?:i live in|residing in|currently living in|based in|from)\s+([A-Za-z\s]+?)(?:[.,;]|\s+and|\s+to|$)"
    ]
    study_patterns = [
        r"(?:planning to study in|wish to study in|want to study in|desire to study in)\s+([A-Za-z\s]+?)(?:[.,;]|\s+and|\s+to|$)"
    ]

    for pattern in residence_patterns:
        match = re.search(pattern, email_text, re.IGNORECASE)
        if match:
            countries["Country of Residence"] = match.group(1).strip()

    for pattern in study_patterns:
        match = re.search(pattern, email_text, re.IGNORECASE)
        if match:
            countries["Desired Country for Studies"] = match.group(1).strip()

    if not countries["Country of Residence"]:
        countries["Country of Residence"] = next((entity for entity in country_entities if "study" not in entity.lower()), None)
    if not countries["Desired Country for Studies"]:
        countries["Desired Country for Studies"] = next((entity for entity in country_entities if "study" in entity.lower()), None)

    return countries

def extract_email_features(email_text, doc):
    """Extract features from the email text."""
    features = {}

    name_match = re.search(r"(?:my name is|this is|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", email_text, re.IGNORECASE)
    features["Name"] = name_match.group(1) if name_match else None

    phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}", email_text)
    features["Phone Number"] = phone_match.group(0) if phone_match else None

    features["Desired Country for Studies"] = process.extractOne(email_text, COUNTRIES)[0] if process.extractOne(email_text, COUNTRIES)[1] > 50 else None

    countries = extract_countries(email_text, doc)
    features["Country of Residence"] = countries["Country of Residence"]

    features["Highest Qualification Completed"] = extract_highest_qualification(email_text, doc)
    features["Intended Field of Study or Program"] = get_field_of_study(email_text)

    start_date_match = re.search(
        r"(?:start(?:ing)?|intake|joining|beginning|planning to start)\s+(?:in\s+)?((?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b(?:\s+\d{4})?)|(?:Fall|Spring|Summer|Winter)\s+\d{4})",
        email_text, re.IGNORECASE
    )
    features["Intake/Start Date"] = start_date_match.group(1).strip() if start_date_match else None

    sponsor_match = re.search(r"(?:sponsor(?:ship)?|fund(?:ing)?|loan|scholarship|accommodation)", email_text, re.IGNORECASE)
    features["Sponsor/Scholarship Request"] = sponsor_match.group(0).strip() if sponsor_match else None

    features["Proficiency Test"], features["Proficiency Test Actual Score"] = extract_tests_scores(email_text, doc)

    return features

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

def match_features_to_model_columns(features):

    feature_dict = {
        "student_country": [features.get('Country of Residence')],
        "preferred_contact_time": [features.get('Preferred Contact Time')],
        "preferred_university": [features.get('Preferred University')],
        "highest_qualification": [features.get('Highest Qualification Completed')],
        "destination_country": [features.get('Desired Country for Studies')],
        "area_of_study": [features.get('Intended Field of Study or Program')],
        "intake": [get_intake_with_year(features.get('Intake/Start Date'))],
        "budget": [features.get('Budget for Tuition and Living Expenses')],
        "english_test": [features.get('Proficiency Test')],
        "english_test_score": [features.get('Proficiency Test Score')],
        "time_to_study": [months_between_dates(features.get('Intake/Start Date'))],
        "intro_source": [features.get('How do you know about us?')],
        "app_used": ["No"],
        "sponsor": [features.get('Sponsor/Scholarship Request')],
        "event_attended": ["No"],
        "standardized_test_Score": [features.get('Proficiency Test Score')]  # New feature
    }

    return feature_dict


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


def store_prediction(features, email, prediction):

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
        "email": email,  # New email field
        "features": features,  # Storing all features in a Map
        "prediction": prediction,
        "timestamp": timestamp_ist  # Store timestamp

    }
    # Insert into DynamoDB
    Predictions_table.put_item(Item=item)
    print(f"Stored Prediction ID: {prediction_id}")

# def handler(event, context):

    
#     for record in event['Records']:
#         message = json.loads(record['body'])['Message']
#         print(f'Received message: {json.loads(message)}')

#         message_attributes = record.get("messageAttributes", {})

#         # Access specific attributes
#         if "Source" in message_attributes:
#             source = message_attributes["Source"]["Value"]
            
#             email = message.get('email')

#         if source == "email":
#             if predict_intent(email) == 'Study Abroad Inquiry':
                
#                 doc = nlp(email)

#                 features= extract_email_features(email,doc)
#                 xgboost_model,preprocessor =  download_models()

#                 data_dict = match_features_to_model_columns(features)

#                 # Convert dictionary to DataFrame
#                 input_df = pd.DataFrame(data_dict)

#                 # Transform the input data using the preprocessor
#                 input_transformed = preprocessor.transform(input_df)

#                 # Get prediction from the model
#                 prediction = xgboost_model.predict(input_transformed)

#                 # Print the result
#                 print("Predicted Target:", prediction[0])

#                 store_prediction(features, prediction[0])
        
#         else:
#             print(f"Unknown source - {source}")
        

def handler():

    
    message = {"email_body":"""Hello, my name is Mark Thompson. I am currently residing in the United States and planning to pursue a Master's in Data Science in Australia.
    I have completed my bachelor's in Computer Science and will be starting my studies in Fall 2025. My phone number is +1212555 9876.
    Could you please guide me regarding the scholarship opportunities available? I have scored 105 in GMAT exam.""","email":"user1@gmaik.com"}

    # Access specific attributes
        
    email = message.get('email_body')

    source= 'email'

    if source == "email":
        intent = predict_intent(email)
        print(intent)
        if intent == 'Study Abroad Inquiry':

            # print("Intent")
            
            
            doc = nlp(email)

            print("Extracting features")
            features= extract_email_features(email,doc)

            print(features)

            print("Downloading models")
            xgboost_model,preprocessor =  download_models()

            print("Matching features")
            data_dict = match_features_to_model_columns(features)
            print(data_dict)

            
            # Convert dictionary to DataFrame
            input_df = pd.DataFrame(data_dict)

            # Transform the input data using the preprocessor
            input_transformed = preprocessor.transform(input_df)

            print("Prediction")
            # Get prediction from the model
            prediction = xgboost_model.predict(input_transformed)

            # Print the result
            print("Predicted Target:", prediction[0])

            print("Storing prediction")

            
            store_prediction(features,message.get('email'),prediction[0])

            print("prediction stored")
    
    else:
        print(f"Unknown source - {source}")
    

handler()