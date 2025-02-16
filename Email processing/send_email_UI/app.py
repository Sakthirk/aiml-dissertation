from flask import Flask, render_template, request, jsonify
import boto3
import json

app = Flask(__name__)

sns_client = boto3.client('sns', region_name="us-east-1")
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:337909783829:email-notification-topic"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        data = {
            "default": json.dumps({
                "name": request.form.get('name', ''),
                "phone_number": request.form.get('phone_number', ''),
                "country_of_residence": request.form.get('country_of_residence', ''),
                "preferred_university": request.form.get('preferred_university', ''),
                "highest_qualification": request.form.get('highest_qualification', ''),
                "preferred_contact_time": request.form.get('preferred_contact_time', ''),
                "desired_country": request.form.get('desired_country', ''),
                "field_of_study": request.form.get('field_of_study', ''),
                "start_date": request.form.get('start_date', ''),
                "budget": request.form.get('budget', ''),
                "proficiency_test": request.form.get('proficiency_test', ''),
                "proficiency_test_score": request.form.get('proficiency_test_score', ''),
                "how_you_know": request.form.get('how_you_know', ''),
                "sponsor": request.form.get('sponsor', '')
            })
        }

        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(data),
            MessageStructure="json",
            MessageAttributes={
                'Source': {
                    'DataType': 'String',
                    'StringValue': "form"
                }
            }
        )

        return jsonify({"success": True, "message_id": response['MessageId']})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/send_email', methods=['POST'])
def send_email():
    try:

        email_data = {
            "default": json.dumps({
                "email_message": request.form['email_message']
            })
        }


        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(email_data),
            Subject="Custom Email",
            MessageStructure="json",
            MessageAttributes={
                'Source': {
                    'DataType': 'String',
                    'StringValue': "email"
                }
            }
        )

        return jsonify({"success": True, "message_id": response['MessageId']})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
