from xgboost import XGBClassifier, cv, DMatrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import re
import boto3
import numpy as np
from io import StringIO
import joblib


s3 = boto3.client('s3')

def read_data(s3_bucket,training_data_path):

    response = s3.get_object(Bucket=s3_bucket, Key=training_data_path)
    
    # Convert response to a DataFrame
    df = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
    return df


def standardize_score(row):
    if row["english_test"] == "IELTS":
        # IELTS: Original scale 0-9
        return 1 + (row["english_test_score"] - 0) * (10 - 1) / (9 - 0)
    elif row["english_test"] == "TOEFL":
        # TOEFL: Original scale 0-120
        return 1 + (row["english_test_score"] - 0) * (10 - 1) / (120 - 0)
    else:
        return None  # Handle unexpected test types


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

def rename_columns(df):
  new_column_names = {
      'Country of Residence': 'student_country',
      'Preferred University': 'preferred_university',
      'Highest Qualification Completed': 'highest_qualification',
      'Desired Country for Studies': 'destination_country',
      'Intended Field of Study or Program': 'area_of_study',
      'Intake/Start Date': 'intake',
      'Budget for Tuition and Living Expenses': 'budget',
      'Proficiency Test': 'english_test',
      'Proficiency Test Actual Score': 'english_test_score',
      'Time to Study (Months)': 'time_to_study',
      'How do you know about us?': 'intro_source',
      'App Used Prior': 'app_used',
      'Sponsor':'sponsor',
      "Any Event Attended":'event_attended',
      "Application Status":'target',
      "Preferred Contact Time":"preferred_contact_time"
  }

  df = df.rename(columns=new_column_names)

  return df



def pre_processing(df):

  df = rename_columns(df)

  # Remove unwanted columns
  columns_to_remove = ['Name', 'Phone Number', 'Prior English Test Booking', 'score']
  df = df.drop(columns=columns_to_remove, errors='ignore')

  # Replace values in 'target' column
  df['target'] = df['target'].replace({'Application Submitted': 1, 'Application Not Submitted': 0})

  # Standardize test score
  df['english_test_score'] = df['english_test_score'].replace({np.nan: 0.0})
  df["standardized_test_Score"] = df.apply(standardize_score, axis=1)

  # Preprocess budget and extract actual budget value
  df['budget'] = df['budget'].apply(process_budget)

  return df


def upload_file(file,bucket_name, s3_key):
    
    s3.upload_file(file, bucket_name, s3_key)
    
    return f"Model uploaded to s3://{bucket_name}/{s3_key}"



def train_and_upload():

    
    # AWS S3 bucket details
    s3_bucket = "aiml-dissertation"
    
    training_data_s3_key = "training_data/mockup_labeled_data.csv"  # Replace with the actual S3 key (file path)

    print('Read data from s3')

    # Step 1: Read data
    data = read_data(s3_bucket,training_data_s3_key)

    print('Pre-processing the data')
    
    # Step 2: Pre-processing
    data = pre_processing(data)

    print('Pre-processing completed')

    # Step 3: Define feature columns (X) and target column (y)
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target variable

    # Step 4: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Preprocessing the features
    categorical_features = ['student_country', 'preferred_contact_time', 'preferred_university', 'highest_qualification', 'destination_country', 'area_of_study', 'intro_source', 'app_used', 'sponsor', 'event_attended']
    numerical_features = ['english_test_score', 'budget', 'time_to_study']

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Step 6: Define the parameter grid
    param_grid = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_lambda': 1,
        'objective': 'binary:logistic'
    }

    # Convert your training data to DMatrix
    dtrain = DMatrix(X_train, label=y_train)

    # Step 7: Perform cross-validation
    cv_results = cv(
        params=param_grid,
        dtrain=dtrain,
        num_boost_round=100,
        nfold=5,
        metrics='logloss',
        as_pandas=True,
        seed=42
    )


    # Step 8: Train the model with the best parameters
    best_num_boost_round = cv_results['test-logloss-mean'].idxmin()
    best_xgb_model = XGBClassifier(
        n_estimators=best_num_boost_round,
        max_depth=param_grid['max_depth'],
        learning_rate=param_grid['learning_rate'],
        subsample=param_grid['subsample'],
        colsample_bytree=param_grid['colsample_bytree'],
        gamma=param_grid['gamma'],
        reg_lambda=param_grid['reg_lambda'],
        objective=param_grid['objective'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print('Fit the model')

    best_xgb_model.fit(X_train, y_train)

    # Save the trained model to a joblib file
    joblib.dump(best_xgb_model, "xgb_model.joblib")

    # Save the preprocessor (so you can use it later for inference)
    joblib.dump(preprocessor, "preprocessor.joblib")

    print('Upload file to s3')
    upload_file("xgb_model.joblib",s3_bucket, "trained_model/xgb_model.joblib")
    upload_file("preprocessor.joblib",s3_bucket, "trained_model/preprocessor.joblib")

def lambda_handler(event, context):
    print("Training started")
    train_and_upload()
    print("Training Completed")


