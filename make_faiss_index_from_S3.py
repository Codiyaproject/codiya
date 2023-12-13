# =============================================================================
# s3 아우터 이미지 목록
# =============================================================================
'''
    S3 -> 파일 하나씩 가져와서 특성 추출후  
    faiss 벡터 디비 ( index ) 에 넣는 코드.
    S3 는 폴더 구분없이 파일명으로 정렬 되어있으며 
    fiass 
'''




import boto3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import pandas as pd 
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
import faiss
import PIL
import os
import cv2
from tqdm import tqdm

# mysql 계정 정보 가져오기 ==================
import pymysql

mysql = {"host" : "fproject-db.c0z1pznbpt0r.us-east-2.rds.amazonaws.com",
"port" : 3306,
"db" : "fashion",
"user" : "codiya",
"password" : "fproject123!"
}

HOST, PORT, DB, USER, PASSWORD = mysql.values()

# DB 연결 하기
def connection():
    try:
        con =  pymysql.connect(
                    host = HOST,
                    port = PORT,
                    db = DB,
                    user = USER,
                    password = PASSWORD
                )
        return con
    except pymysql.Error:
        print(pymysql.Error)

#         
def db_img_name(table):
    # tables = ["musinsa_outer", "musinsa_top", "musinsa_onepiece", "musinsa_bottom"]
    data = list()
    con = connection()
    cur = con.cursor()
    sql = f"SELECT img_path FROM {table} order by id"
    cur.execute(sql)
    for path in cur.fetchall():
        data.append(path[0])
    return data

connection()
print(db_img_name("musinsa_outer")) # 파일명 리스트 가져오기 

#===============================================================================
# AWS 계정 정보
ACCESS_KEY_ID = 'AKIA2S5N3ADHYPR3Y7MJ'
ACCESS_SECRET_KEY = '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'
BUCKET_NAME = 'fproject-codiya'

# S3 클라이언트 생성
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=ACCESS_SECRET_KEY)

# 객체 목록을 담을 리스트
all_objects = []

# 첫 페이지 요청
response = s3.list_objects_v2(Bucket=BUCKET_NAME)

# 첫 페이지의 객체를 리스트에 추가
all_objects.extend(response.get('Contents', []))

# 다음 페이지 요청
while response.get('IsTruncated', False):
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, ContinuationToken=response['NextContinuationToken'])
    all_objects.extend(response.get('Contents', []))

# s3 에 있는 전체 파일 목록    
# s3_file_list = list()
# for obj in all_objects:
#     s3_file_list.append(obj['Key'])

# # pandas 로 csv 에 있는 순서대로 파일명 가져오기
# pants = pd.read_csv('/content/drive/MyDrive/프로젝트/finetuning_model1/Outer_final.csv')#,encoding='cp949'
# s3_pants_list = pants['file_name_jpg'].tolist()

# 벡터디비를 만들 리스트 가져오기 ( id 순서임 )
# 
tables = ["musinsa_outer", "musinsa_top", "musinsa_onepiece", "musinsa_bottom"]
for table in tables : 
    table = db_img_name(table)

print(len("musinsa_outer"))
# 43234u32432.jpg
# =============================================================================
# s3 이미지 보기
# =============================================================================
import boto3
import io
import matplotlib.pyplot as plt
from PIL import Image


# ## 훈련 모델 가져오기
# base_model = keras.models.load_model('fine_tunning_model_ver1\\fine_tuning_model.h5')
# # EfficientNetV2S 모델의 데이터 추출 모드 사용 -> 이미지 특성 모델
# similar_model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    


# def extract_features(image_key, similar_model):

#     response = s3.get_object(Bucket=BUCKET_NAME, Key=image_key)
#     image_data = response['Body'].read()
    
#     img = image.load_img(io.BytesIO(image_data), target_size=(512, 512))  # 
#     #img.save("extract_features_test.jpeg") 
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features = similar_model.predict(x)
#     flatten_features =  features.flatten()

#     return flatten_features



# # 벡터디비 생성
# def db_create(model):
#     # Initialize FAISS index
#     dimension = model.output.shape[1]  # This should match the feature vector size
#     index = faiss.IndexFlatL2(dimension)
#     return index

# # 모델 생성

# # DB 생성
# index = db_create(similar_model)

# # faiss index 에 담는 함수 
# def make_faiss_index(category) : 
#     category = pd.read_csv(f'musinsa_csv\{category}_final_delete_jpg.csv')#,encoding='cp949'
#     s3_file_list = category['file_name_jpg'].tolist()
#     for image_key in tqdm(s3_file_list):  # dataset_paths is a list of image file paths
#             features = extract_features(image_key, similar_model)
#             index.add(np.array([features]))  # Add features to FAISS index
            
#     faiss.write_index(index, "S3_outer_index_test.faiss")

# make_faiss_index("Outer")

# # print("인덱스 길이 :" ,len)

