import boto3
import numpy as np
import logging
from scipy.stats import ks_2samp
import sys

# Configure logging
logger = logging.getLogger()
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)  # Force logs to stdout
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)  # Force debug level

# Initialize AWS Clients
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client("sqs")

# Table references
prediction_table = dynamodb.Table('Predictions')
actuals_table = dynamodb.Table('application')

SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/337909783829/model-training-queue"



def fetch_predictions():
    """Fetch predicted probabilities from DynamoDB."""
    try:
        response = prediction_table.scan()
        predictions = {
            item['email']: float(item.get('predicted_probability', '0.5').strip('[]'))
            for item in response.get('Items', [])
        }
        logger.info(f"Fetched {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        return {}

def fetch_actuals():
    """Fetch actual submission records from DynamoDB."""
    try:
        response = actuals_table.scan()
        actuals = {item['email_id']: 1 for item in response.get('Items', [])}
        logger.info(f"Fetched {len(actuals)} actual records")
        return actuals
    except Exception as e:
        logger.error(f"Error fetching actuals: {str(e)}")
        return {}

def bin_probabilities(probabilities, bins=4):
    """Bins probabilities into discrete categories."""
    return np.histogram(probabilities, bins=np.linspace(0, 1, bins + 1))[0]

def detect_model_drift(pred_probs, actual_labels, threshold=0.1):
    """Detects drift by comparing binned predicted probabilities with actual submission rates."""
    common_emails = set(pred_probs.keys()).intersection(set(actual_labels.keys()))

    if not common_emails:
        logger.warning("No common email IDs found! Skipping drift detection.")
        return False

    predicted_values = [pred_probs[email] for email in common_emails]
    actual_values = [actual_labels[email] for email in common_emails]

    if not predicted_values or not actual_values:
        logger.warning("One of the distributions is empty! Cannot perform KS test.")
        return False

    # Bin both distributions
    pred_bins = bin_probabilities(predicted_values)
    actual_bins = bin_probabilities(actual_values)

    logger.info(f"Binned Predictions: {pred_bins}")
    logger.info(f"Binned Actuals: {actual_bins}")

    # Perform Kolmogorov-Smirnov Test
    ks_stat, p_value = ks_2samp(pred_bins, actual_bins)
    logger.info(f"KS Statistic: {ks_stat}, P-Value: {p_value}")

    return ks_stat > threshold

def send_sqs_message():
    """Sends a message to SQS when model drift is detected."""
    try:
        response = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody="Model drift detected! Retraining required."
        )
        logger.info(f"Message sent to SQS. Message ID: {response['MessageId']}")
    except Exception as e:
        logger.error(f"Error sending message to SQS: {str(e)}")

def lambda_handler(event, context):
    """Lambda entry point to check model drift and trigger retraining if necessary."""
    logger.info("Starting model drift detection...")

    predictions = fetch_predictions()
    actuals = fetch_actuals()

    matched = len(set(predictions.keys()).intersection(set(actuals.keys())))
    unmatched = len(predictions) - matched
    logger.info(f"Matched records: {matched}, Unmatched predictions: {unmatched}")

    model_drift_detected = detect_model_drift(predictions, actuals)

    if model_drift_detected:
        logger.info("Drift detected! Sending SQS message...")
        send_sqs_message()
        return {"status": "Drift detected, retraining triggered"}

    logger.info("No drift detected.")
    return {"status": "No drift detected"}