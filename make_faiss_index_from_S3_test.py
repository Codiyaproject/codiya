# =============================================================================
# s3 아우터 이미지 목록
# =============================================================================
'''
    S3 -> 파일 하나씩 가져와서 특성 추출후  
    faiss 벡터 디비 ( index ) 에 넣는 코드.
    S3 는 폴더 구분없이 파일명으로 정렬 되어있으며 
    db 에서 파일명 리스트를 가져온 후 , 그 리스트 순서에 맞게 벡터db 생성됨. 
    
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

# 파일명 리스트 가져오기 
def db_img_name(table):
    data = list()
    con = connection()
    cur = con.cursor()
    sql = f"SELECT img_path FROM {table} order by id"
    cur.execute(sql)
    for path in cur.fetchall():
        data.append(path[0])
    return data

connection()

# S3 연결 ===============================================================================
# AWS 계정 정보

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
# ==============================================================

# sql 에 있는 테이블의 파일명 가져오기 ( id 순서 ) ===========

tables = ["musinsa_outer" , "musinsa_bottom"]# , "musinsa_onepiece" , "musinsa_top" ] # 일단 top 은 안가져옴
for table in tables : 
    globals()[f'{table}_list'] = db_img_name(table)

print("무신사아우터:" , len(musinsa_outer_list))
print("무신사바텀:" , len(musinsa_bottom_list))
# ==================================================

import boto3
import io


## 훈련 모델 가져오기 ====================
base_model = keras.models.load_model('fine_tunning_model_ver1\\fine_tuning_model.h5')
similar_model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
# ==============================


def extract_features(image_key, similar_model):

    response = s3.get_object(Bucket=BUCKET_NAME, Key=image_key)
    image_data = response['Body'].read()
    
    img = image.load_img(io.BytesIO(image_data), target_size=(512, 512))  # 
    #img.save("extract_features_test.jpeg") 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = similar_model.predict(x)
    flatten_features =  features.flatten()

    return flatten_features



# 벡터디비 생성
def db_create(model):
    dimension = model.output.shape[1]  
    index = faiss.IndexFlatL2(dimension)
    return index

# 모델 생성

# DB 생성
index = db_create(similar_model)
file_list_list=[musinsa_outer_list,musinsa_bottom_list]#,musinsa_onepiece_list,musinsa_top_list]
categories = ['outer','bottom','onepiece','top']
#,onepiece_index,top_index]
# faiss index 에 담는 함수 
def make_faiss_index(categories) : 
    
    for i in range(len(categories)) : 

        for image_key in tqdm(file_list_list[i]):  # dataset_paths is a list of image file paths
                globals()[f"{categories[i]}_index"] = db_create(similar_model)
                features = extract_features(image_key, similar_model)
                index.add(np.array([features]))  # Add features to FAISS index
    
        index_list = [outer_index,bottom_index]            
        index_list[i].write_index(index, f"S3_{categories[i]}_index_test.faiss")

make_faiss_index(categories)

# print("인덱스 길이 :" ,len)

