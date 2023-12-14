import json

with open("/home/ubuntu/key.json", "r") as f:
        KEY = json.load(f)
        
def s3_config():    
    ACCESS_KEY_ID = KEY["S3_KEY"]["ACCESS_KEY_ID"]
    ACCESS_SECRET_KEY = KEY["S3_KEY"]["ACCESS_SECRET_KEY"]        
    BUCKET_NAME = KEY["S3_KEY"]["BUCKET_NAME"]
    REGION = KEY["S3_KEY"]["REGION"]
    return ACCESS_KEY_ID, ACCESS_SECRET_KEY, BUCKET_NAME, REGION


def mysql_config():
    host = KEY["SQL_KEY"]["host"]
    port = KEY["SQL_KEY"]["port"]
    db = KEY["SQL_KEY"]["db"]
    user = KEY["SQL_KEY"]["user"]
    password = KEY["SQL_KEY"]["password"] 
    return host, port, db, user, password

def openai_config():
    OPENAI_API_KEY = KEY["OPENAI_API_KEY"]
    return OPENAI_API_KEY

config = KEY["config"]