import boto3
import json
import uuid
import datetime
import os
from decimal import Decimal

sagemaker_client = boto3.client("sagemaker-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("InferenceLogs")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]


def lambda_handler(event, context):
    # Get the input text from the API Gateway event

    # Handle direct invocation (like HTTP API or test event)
    if isinstance(event, dict) and "text" in event:
        input_text = event["text"]
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'text' in request"}),
        }

    # Call SageMaker endpoint
    response = sagemaker_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": input_text}),
    )

    prediction = json.loads(response["Body"].read().decode())
    label = prediction[0].get("label")
    score = prediction[0].get("score")

    # Convert score to Decimal
    score = Decimal(str(score))

    # Generate a UUID and timestamp
    request_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()

    # Log to DynamoDB
    table.put_item(
        Item={
            "request_id": request_id,
            "timestamp": timestamp,
            "input_text": input_text,
            "label": label,
            "score": score,
        }
    )

    # Return the prediction
    return {
        "statusCode": 200,
        "body": json.dumps({"request_id": request_id, "prediction": prediction}),
    }
