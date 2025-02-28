import json
import logging
from email_processing import EmailProcessing
from form_processing import FormProcessing
import sys


logger = logging.getLogger()
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)  # Force logs to stdout
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)  # Force debug level

def lambda_handler(event, context):

        try:
            logging.info(f'Received event: {event}')
            
            for record in event['Records']:
                message_body = json.loads(record['body'])
                message = json.loads(message_body['Message'])
                message_attributes = message_body.get("MessageAttributes", {})  # Message attributes
                source = message_attributes.get("Source", {}).get("Value", "N/A")

            
                if source == 'email':
                     email_processing = EmailProcessing()
                     rtn = email_processing.lambda_handler(message)
                elif source == 'form':
                     form_processing = FormProcessing()
                     rtn = form_processing.lambda_handler(message)
                else:
                     rtn = f"Unknown payload with source - {source}"
                
                logging.info(rtn)

                     
        except Exception as error:
            logging.info("Exception raised")
            logging.error(error)
            raise Exception(error)