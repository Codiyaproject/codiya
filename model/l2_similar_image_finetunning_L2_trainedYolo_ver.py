# 모듈 불러오기
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
import keras
from roboflow import Roboflow
from PIL import Image
import numpy as np
import faiss
import PIL
import os
import cv2
import csv
import urllib.request 
import requests
import random
import pandas as pd 
import boto3
import io
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import json
from config import mysql_config , s3_config
import pymysql

## S3 환경 불러오기 ========================================================
HOST, PORT, DB, USER, PASSWORD = mysql_config()

# AWS 계정 정보

ACCESS_KEY_ID , ACCESS_SECRET_KEY , BUCKET_NAME = s3_config()

import boto3 

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
   
s3_file_list = list()
for obj in all_objects:
    s3_file_list.append(obj['Key'])
# S3 ==============================================================================================
# # pandas 로 csv 에 있는 순서대로 파일명 가져오기
# pants = pd.read_csv('/content/drive/MyDrive/프로젝트/finetuning_model1/Outer_final.csv')#,encoding='cp949'
# s3_pants_list = pants['file_name_jpg'].tolist()
# ========================================================================
  


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


# csv 에 먼저 접근 후 파일 경로 가져오기 ## ---> sql 로 변경 
# tables = ["musinsa_outer", "musinsa_top", "musinsa_onepiece", "musinsa_bottom"]
bottom_s3_file_list = db_img_name('musinsa_bottom')

Onepiece_pd = pd.read_csv(f'musinsa_csv\\Onepiece_sample_test.csv',encoding='cp949')#,encoding='cp949'
onepiece_s3_file_list = db_img_name('musinsa_onepiece')

Outer_pd = pd.read_csv(f'musinsa_csv\Outer_final_delete_jpg.csv')#,encoding='cp949' <에러날시 인자에 추가
outer_s3_file_list = db_img_name('musinsa_outer')   


# 테스트로 1개만 만들었음  

# category_pd = pd.read_csv(f'musinsa_csv\outer_final_delete_jpg.csv')#,encoding='cp949'
# outer_s3_file_list = category_pd['file_name_jpg'].tolist()


# 패션이미지 category 분류 모델
def predict_yolo(trained_yolo, img_path, confidence=0.4):
    '''
    return : x, y 값 좌표 ( 예측된 사각형의 젤 왼쪽 위 끝점 임 )
            width , height
            class : 카테고리 이름
    '''
    # 이미지 불러오기 ( 나중에 자르기 위해서  )
    image = cv2.imread(img_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Yolo 모델로 이미지 예측하기
    predict_info = trained_yolo.predict(img_path, conf = confidence, save = False)
    #predict.save("yolo_test.jpg")
    # 예측 정보 출력
    
    # 잘 탐지했나 저장해서 테스트
    #predict.save("save_test.jpeg") 
    # 예측 이미지 저장 공간 생성
    yolo_result = list()
    # 이미지 카테고리 . # 
    result_categories = list()  # 이미지 인덱스 = 카테고리 인덱스 

    # 이미지 예측된 부위 데이터 담기
    
    for info in predict_info : # 탐지한 전체 정보중 
        if len(info) == 0 : # 만약 카테고리가 비어있다면  
            result_categories = [0,1,2]
            yolo_result.append(image[:, :, :])
            yolo_result.append(image[:, :, :])
            yolo_result.append(image[:, :, :])
        else :
            for cord in info :                    
                x,y,w,h = cord[0].boxes.xywh.cpu().numpy()[0]
                x,y,w,h = int(x),int(y),int(w) , int(h)
                start_x, start_y, width, height , category = (x - round(w/2) - 3) , (y - round(h/2) - 3) ,w , h , int(cord[0].boxes.cls.cpu().numpy()[0])
                end_x, end_y = (start_x + width), (start_y + height)
                yolo_result.append(image[start_y:end_y, start_x:end_x, :]) 
                result_categories.append(category)
    #잘 잘렸나 저장해서 테스트
    print("yolo_result" , len(yolo_result)) 
    #print(yolo_result) 

  
                
    im = Image.fromarray(yolo_result[0])        
    im.save(f"yolotest_.jpeg")  
    
    return yolo_result, result_categories # yolo_result : 이미지 형식 . 잘라놓은 것 

# 모델 생성하기 
def model_create():
    # EfficientNetV2S 모델 객체 생성
    base_model = keras.models.load_model('fine_tunning_model_ver1\\fine_tuning_model.h5')
    # EfficientNetV2S 모델의 데이터 추출 모드 사용 -> 이미지 특성 모델
    similar_model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    
    # Roboflow 모델 생성 -> 이미지 식별 모델
    trained_yolo = YOLO('yolo_trained_model.pt')
    # rf = Roboflow(api_key="Q78HnDQgOoukAA6rXrsG")
    # project = rf.workspace().project("fashion-hkjfr")
    # rf_model = project.version(5).model
    return similar_model, trained_yolo





# 이미지 크기 조정(input data) 만들기
def extract_features(img, model):
    #img_path = "/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/" + img_path
    # img = image.load_img(img_path, target_size=(384, 384))
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    img = Image.fromarray(img).resize((512, 512))
    #img.save("extract_features_test.jpeg") 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    #print(features)
    return features.flatten()

# 생성된 이미지 category 분류 하기 유사 이미지 추천해주기
def search_similar_images(query_img_path, similar_model, rf_model, n_results=3):
    """
          카테고리 분류 : 'jacket' , 'shirt' , 'pants' ,' skirt' , 'dress'
                            = 0 ,       =1       =2        =3        =4 
    """
    result = [] # 파일 이름이 나와야함 
    # 욜로로 탐지해서 이미지 카테고리와 해당하는 잘라진 이미지 획득   
    pred_img, pred_category  = predict_yolo(rf_model, query_img_path) 
    print("pred_카테고리 : " , pred_category)
    for img, category in zip(pred_img, pred_category): 
        print(len(pred_img))
        match category:

            # case 0: # jacket 
            #     print('jacket case 문 시작')
            
                
            #     query_features = extract_features(img, similar_model)

            #     distances, indices = bottom_index.search(np.array([query_features]), n_results)
            #     # Retrieve the similar images
            #     similar_images = [bottom_index[i] for i in indices[0]]
            #     result += similar_images
                
                
            case 1 : # shirts
                print('shirts case 문 시작')
                query_features = extract_features(img, similar_model)
                
                # Search the FAISS index
                #print( " chat : " , query_features)
                distances, indices = onepiece_index.search(np.array([query_features]), n_results)
                # Retrieve the similar images
                similar_images = [onepiece_s3_file_list[i] for i in indices[0]]   
                result += similar_images

                                    
            case 2: # pants
                print('pants case 문 시작')
                
                query_features = extract_features(img, similar_model)

                # Search the FAISS index
                #print( " chat : " , query_features)
                distances, indices = onepiece_index.search(np.array([query_features]), n_results) ## 현재 outer 제외 준비된 데이터가 없어서 모두  outer 로 처리하였음
                print("거리 : " , distances)
                # Retrieve the similar images
                similar_images = [onepiece_s3_file_list[i] for i in indices[0]]
                result += similar_images
                
            case 3: # pants
                print('pants case 문 시작')
                
                query_features = extract_features(img, similar_model)

                # Search the FAISS index
                #print( " chat : " , query_features)
                distances, indices = onepiece_index.search(np.array([query_features]), n_results) ## 현재 outer 제외 준비된 데이터가 없어서 모두  outer 로 처리하였음
                print("거리 : " , distances)
                # Retrieve the similar images
                similar_images = [onepiece_s3_file_list[i] for i in indices[0]]
                result += similar_images                
                
                
            case 4: # onepiece 
                
                query_features = extract_features(img, similar_model)

                # Search the FAISS index
                #print( " chat : " , query_features)
                distances, indices = onepiece_index.search(np.array([query_features]), n_results)
                # Retrieve the similar images
                similar_images = [onepiece_s3_file_list[i] for i in indices[0]]
                result += similar_images
                
                
    return result , pred_category           
                    
    
    # for img in pred_img:
    #     query_features = extract_features(img, similar_model)

    #     # Search the FAISS index
    #     print( " chat : " , query_features)
    #     distances, indices = index.search(np.array([query_features]), n_results)
    #     # Retrieve the similar images
    #     similar_images = [s3_file_list[i] for i in indices[0]]
    #     result += similar_images
    # return result , pred_category

# 벡터 디비 호출
outer_index = faiss.read_index("S3_outer_index_test.faiss")
bottom_index = faiss.read_index("index_L2_bottom_sample.faiss")
onepiece_index = faiss.read_index("index_L1_onepiece_sample.faiss")
# 모델 생성
similar_model, rf_model = model_create()
        
# 테스트 파일 
# image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-isiflj0QPpZOqxTrr3DNR0q8/user-LT48mRZarJXcn0PeAjmcuwrc/img-GkhndvGEPuwVKSqBACRGNrKD.png?st=2023-11-28T03%3A34%3A00Z&se=2023-11-28T05%3A34%3A00Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-28T02%3A51%3A55Z&ske=2023-11-29T02%3A51%3A55Z&sks=b&skv=2021-08-06&sig=GrXt2dc0esy8bFLD7%2BBM2g82nO8X2dLGIPncSKw1TsA%3D"

# res = requests.get(image_url)
# img = Image.open(BytesIO(res.content))
# img
query_img_path = 'data_storage\\target_data\\top\\blue_jean_tsurt _top.png'
similar_images,pred_category = search_similar_images(query_img_path, similar_model,rf_model, n_results = 3) 
print("상품 이름" , similar_images) 
print("카테고리 : ", pred_category)


# 유사한 이미지 출력
for img_name in similar_images: 
    response = s3.get_object(Bucket=BUCKET_NAME, Key=img_name)
    image_data = response['Body'].read()

    # BytesIO를 사용하여 이미지 데이터를 PIL Image로 변환
    # 이 이미지를 백터화 작업 진행하면 됨
    image = Image.open(io.BytesIO(image_data))
    
    
    # 이 코드는 사용하지 않아도 됨
    # 아래의 코드는 이미지를 Matplotlib을 사용하여 이미지 표시하는 코드
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# cv2.imshow 에서 
# cv2.error: OpenCV(4.8.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1266: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvDestroyAllWindows'
# 에러나면 pip uninstall 후 다시  
# !pip uninstall opencv-python
# !pip install opencv-python.

## 사진이 작을경우 yolo 에서 오류나는듯함 
# 사진파일이 한글이면 오류날수도 
# yolo 가 detect 한 카테고리가 없을수도있음 
# 카테고리별로 3개씩 이미지를 띄우지만 outer 에서 4개 , pants 에서 2개 띄울수도있음 
# 만약 사람이 3명 그려졌다면 x3 개가 나옴 
            
