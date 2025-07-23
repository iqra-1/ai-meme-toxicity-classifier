import pandas as pd
import boto3
import io

# Define S3 bucket and prefixes
bucket = "toxic-meme-classification"
input_prefix = "raw_dataset/train.csv"
output_prefix = "feature_dataset/cleaned_train.csv"

# Initialize S3 client
s3 = boto3.client("s3")

def process_and_upload_full_csv():
    # Read full CSV file from S3
    obj = s3.get_object(Bucket=bucket, Key=input_prefix)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    # Create binary label 'y' based on any toxic flag
    df["y"] = (
        df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
        .sum(axis=1) > 0
    ).astype(int)

    # Select and rename relevant columns
    cleaned_df = df[["comment_text", "y"]].rename(columns={"comment_text": "meme_text"})

    # Convert final DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    cleaned_df.to_csv(csv_buffer, index=False)

    # Upload single cleaned CSV to S3
    s3.put_object(Bucket=bucket, Key=output_prefix, Body=csv_buffer.getvalue())
    print(f"Full cleaned dataset uploaded to s3://{bucket}/{output_prefix}")

# Run the process
process_and_upload_full_csv()

