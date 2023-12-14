from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
from DB.s3 import s3
from PIL import Image
import numpy as np
import faiss
import PIL
import cv2
import urllib.request 
from flask import jsonify
import requests
import datetime
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor
import keras
from DB.sql import musinsa_img_name
from ultralytics import YOLO

today = datetime.date.today()
now = datetime.datetime.now()

search_log = f"/web_service/static/log/search/{today}_search.log"

def search_outfit(message):
    
    try:
        with open(search_log,'a') as log_file:
            log_file.write(f"{now}_Outfit : {message}" + '\n')
    except FileNotFoundError:
        with open(search_log,'w') as log_file:
            log_file.write(f"{now}_Outfit : {message}" + '\n')
        
def search_cates(message):
    try:
        with open(search_log,'a') as log_file:
            log_file.write(f"{now}_Cates : {message}" + '\n')
    except FileNotFoundError:
        with open(search_log,'w') as log_file:
            log_file.write(f"{now}_Cates : {message}" + '\n')
        
def search_results(message):
    try:
        with open(search_log,'a') as log_file:
            log_file.write(f"{now}_Results : {message}" + '\n')        
    except FileNotFoundError:
        with open(search_log,'w') as log_file:
            log_file.write(f"{now}_Results : {message}" + '\n')
            

def sequence_end(message):
    try:
        with open(search_log,'a') as log_file:
            log_file.write(f"{message}" + '\n')
    except FileNotFoundError:
        with open(search_log,'w') as log_file:
            log_file.write(f"{message}" + '\n')
            
                   

# SQL 데이터 가져오기
bottom_s3_file_list = musinsa_img_name("musinsa_bottom")

onepiece_s3_file_list = musinsa_img_name("musinsa_onepiece")

top_s3_file_list = musinsa_img_name("musinsa_top")

outer_s3_file_list = musinsa_img_name("musinsa_outer")



bottom_index = faiss.read_index("DB/S3_bottom_L2_index.faiss")
onepiece_index = faiss.read_index("DB/S3_onepiece_L2_index.faiss")
top_index = faiss.read_index("DB/S3_top_L2_index.faiss")
outer_index = faiss.read_index("DB/S3_outer_L2_index.faiss")


def yolo_model_create():
    print("Yolo model_create~~~~~~~~~~")
    
    trained_yolo = YOLO('engine/yolo_trained_model.pt')
    return trained_yolo

def similar_model_create():
    print("Similar model_create~~~~~~~~~~")
    detect_model = keras.models.load_model('engine/fine_tuning_model.h5')
    similar_model = Model(inputs=detect_model.input,outputs=detect_model.get_layer('global_average_pooling2d').output)
    return similar_model

def predict_yolo(title):
    trained_yolo = yolo_model_create()()
    print("predict_yolo~~~~~~~~~~~")
    img_path = "static/images/created_image/" + title + ".png"
    
    predict = trained_yolo.predict(img_path)
    
    dalle_image = cv2.imread(img_path) 
    dalle_image = cv2.cvtColor(dalle_image, cv2.COLOR_BGR2RGB)
    
    yolo_result = list()
    result_cates = list()
    
    for info in predict : 
        if len(info) == 0 :  
            result_cates = ['0','1','2']
            yolo_result.append(image[:, :, :])
            yolo_result.append(image[:, :, :])
            yolo_result.append(image[:, :, :])
        else :
            for cord in info:
                x,y,w,h = cord[0].boxes.xywh.cpu().numpy()[0]
                x,y,w,h = int(x),int(y),int(w),int(h)
                start_x, start_y, width, height , category = (x - round(w/2) - 3) , (y - round(h/2) - 3) , w , h , int(cord[0].boxes.cls.cpu().numpy()[0])
                end_x, end_y = (start_x + width), (start_y + height)
                yolo_result.append(dalle_image[start_y:end_y, start_x:end_x, :]) 
                result_cates.append(category)
    
    return yolo_result, list(set(result_cates))


def extract_features(img,model):
    print("extract_features~~~~~~~~~")
    img = Image.fromarray(img).resize((512,512))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    print("features:",features)
    return features.flatten()


def search_similar_images(title):
    similar_model = similar_model_create()
    print("search_similar_images~~~~~~~~~~")
    n_results = 3
    result = []
    pred_img, pred_category = predict_yolo(title)
    
    for img, category in zip(pred_img, pred_category): 
        match category:
            case 2 :
                query_features = extract_features(img, similar_model)
                distances, indices = bottom_index.search(np.array([query_features]), n_results)
                print("거리 : " , distances)
                similar_images = [bottom_s3_file_list[i] for i in indices[0]]
                result += similar_images
            case 3 :
                query_features = extract_features(img, similar_model)
                distances, indices = bottom_index.search(np.array([query_features]), n_results)
                print("거리 : " , distances)
                similar_images = [bottom_s3_file_list[i] for i in indices[0]]
                result += similar_images
            case 4 :
                query_features = extract_features(img, similar_model)
                distances, indices = onepiece_index.search(np.array([query_features]), n_results)
                print("거리 : " , distances)
                similar_images = [onepiece_s3_file_list[i] for i in indices[0]]
                result += similar_images
                
        
    print("result",result)
    print("pred_category",pred_category)
    
    # for img_name in result: 
    #     response = s3.get_object(Bucket=BUCKET_NAME, Key=img_name)
    #     image_data = response['Body'].read()
    #     print("response : ",response)
    
    search_outfit(title)
    search_cates(pred_category)
    search_results(result)
    sequence_end("====================================================================================")
    
    return jsonify({"result": result, "pred_category": pred_category})