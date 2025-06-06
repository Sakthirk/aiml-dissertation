from flask import Flask, render_template, request, jsonify
import boto3
import json

app = Flask(__name__)

sns_client = boto3.client('sns', region_name="us-east-1")
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:337909783829:customer-segmentation-topic"

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Predictions')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        data = {
            "default": json.dumps({
                "name": request.form.get('name', ''),
                "email": request.form.get('email', ''),
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


@app.route('/update-priority', methods=['POST'])
def update_priority():
    data = request.json
    item_id = data.get('id')
    new_priority = data.get('priority')

    if not item_id or not new_priority:
        return jsonify({'error': 'Missing data'}), 400

    # Update item in DynamoDB
    response = table.update_item(
        Key={'Id': item_id},
        UpdateExpression='SET actual_priority = :priority',
        ExpressionAttributeValues={':priority': new_priority}
    )

    return jsonify({'message': 'Priority updated successfully'})

@app.route('/send_email', methods=['POST'])
def send_email():
    try:

        email_data = {
            "default": json.dumps({
                "email_body": request.form['email_message'],
                "email":request.form['email'],
            })
        }


        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(email_data),
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



@app.route('/get_emails', methods=['GET'])
def get_emails():
    response = table.scan(ProjectionExpression="email")
    emails = list(set(item['email'] for item in response.get('Items', [])))
    return jsonify(emails)

@app.route('/get_submissions', methods=['GET'])
def get_submissions():
    filter_value = request.args.get('filter_value')  # Get filter from request

    if filter_value:
        response = table.scan(
            FilterExpression="email = :email",
            ExpressionAttributeValues={":email": filter_value}
        )
    else:
        response = table.scan()  # No filter, get all records

    print(jsonify(response.get("Items", [])))
    
    return jsonify(response.get("Items", []))

if __name__ == '__main__':
    app.run(debug=True)


# from boto3.dynamodb.conditions import Key

# # Initialize the DynamoDB resource
# # dynamodb = boto3.resource("dynamodb")
# clickstream_table = dynamodb.Table("ClickstreamTable")

# def get_clickstream_data(email_id):
#     response = clickstream_table.query(
#         KeyConditionExpression=Key("email_id").eq(email_id)
#     )
#     return response["Items"]

# # Example usage
# user_email = "user1@gmail.com"
# clickstream_data = get_clickstream_data(user_email)
# print(json.dumps(clickstream_data, indent=4))
