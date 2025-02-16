from flask import Flask, render_template, request, jsonify
import boto3

app = Flask(__name__)

# AWS SNS Client
sns_client = boto3.client('sns', region_name="us-east-1")

# Replace with your SNS Topic ARN
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:337909783829:email-notification-topic"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        message = request.form['message']
        subject = "Message from Web UI"

        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message,
            Subject=subject,
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
