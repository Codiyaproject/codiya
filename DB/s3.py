from config import s3_config
import boto3
from botocore.client import Config
ACCESS_KEY_ID, ACCESS_SECRET_KEY, BUCKET_NAME, REGION = s3_config()

# client 얻기
def get_client():
    return boto3.client(
            service_name = "s3",
            aws_access_key_id = ACCESS_KEY_ID,
            aws_secret_access_key = ACCESS_SECRET_KEY,
            region_name = REGION,
            config=Config(signature_version = 's3v4')
        )
    
def get_resource():
    return boto3.resource(
            service_name = "s3",
            aws_access_key_id = ACCESS_KEY_ID,
            aws_secret_access_key = ACCESS_SECRET_KEY,
            region_name = REGION,
            config=Config(signature_version = 's3v4')
        )

# 이미지 불러오기
def s3_download_img(img_name):
    s3 = get_client()
    return s3.get_object(Bucket = BUCKET_NAME, Key = img_name)

# s3 업로드
def s3_upload_img(file, img_name):
    s3 = get_resource()
    s3.Bucket(BUCKET_NAME).put_object(
        Key = img_name, Body = file, ContentType = "image/jpg"
    )

if __name__ == "__main__":
    # print(get_img("1231285.jpg"))
    pass