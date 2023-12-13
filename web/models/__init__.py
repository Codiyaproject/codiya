from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.models import Model
from roboflow import Roboflow
from PIL import Image
import numpy as np
import faiss
import PIL
import os
import cv2
import csv
import urllib.request 
from flask import jsonify
import requests
import datetime
import logging
import logging.handlers

today = datetime.date.today()
now = datetime.datetime.now()

from concurrent.futures import ThreadPoolExecutor

def search_outfit(message):
    try:
        with open(f'static/log/search/{today}_search.log','a') as log_file:
            log_file.write(f"{now}_Outfit : {message}" + '\n')
    except FileNotFoundError:
        with open(f'static/log/search/{today}_search.log','w') as log_file:
            log_file.write(f"{now}_Outfit : {message}" + '\n')
        
def search_cates(message):
    try:
        with open(f'static/log/search/{today}_search.log','a') as log_file:
            log_file.write(f"{now}_Cates : {message}" + '\n')
    except FileNotFoundError:
        with open(f'static/log/search/{today}_search.log','w') as log_file:
            log_file.write(f"{now}_Cates : {message}" + '\n')
        
def search_results(message):
    try:
        with open(f'static/log/search/{today}_search.log','a') as log_file:
            log_file.write(f"{now}_Results : {message}" + '\n')        
    except FileNotFoundError:
        with open(f'static/log/search/{today}_search.log','w') as log_file:
            log_file.write(f"{now}_Results : {message}" + '\n')
            

def sequence_end(message):
    try:
        with open(f'static/log/search/{today}_search.log','a') as log_file:
            log_file.write(f"{message}" + '\n')
    except FileNotFoundError:
        with open(f'static/log/search/{today}_search.log','w') as log_file:
            log_file.write(f"{message}" + '\n')                   


def ds_create():
    print("ds_create~~~~~~~~~")
    dataset_names = list()
    with open("static/dataset_names.csv",'r') as f:
        file = csv.reader(f)
        for line in file:
            dataset_names += line
    print("len:",len(dataset_names))
    return dataset_names

img_dir = "static/images/data_storage/image"
categoreies = ['onepiece','outer','pants','skirt','top']

def get_file_paths():
    print("get_file_paths~~~~~~~~~~~~")
    file_names = []
    for root,_,filename in os.walk(img_dir):
        if root.split("\\")[-1] in categoreies:
            file_names += filename
            
    return file_names


def model_create():
    print("model_create~~~~~~~~~~")
    rf = Roboflow(api_key="Q78HnDQgOoukAA6rXrsG")
    project = rf.workspace().project("fashion-hkjfr")
    rf_model = project.version(5).model
    
    detect_model = EfficientNetV2S(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    
    similar_model = Model(inputs=detect_model.input,outputs=detect_model.get_layer('avg_pool').output)
    
    return rf_model, similar_model

rf_model, similar_model = model_create()

def predict_yolo(title):
    print("predict_yolo~~~~~~~~~~~")
    img_path = "static/images/created_image/" + title + ".png"
    
    predict = rf_model.predict(img_path)
    
    dalle_image = cv2.imread(img_path) 
    dalle_image = cv2.cvtColor(dalle_image, cv2.COLOR_BGR2RGB)
    
    
    predicts = predict.json()
    yolo_result = list()
    result_cates = list()
    
    for predict in predicts["predictions"]:
        if predict["class"] not in ["hat","sunglass","bag","shoe"]:
            start_x, start_y, width, height, category = predict['x']-(predict["width"]//2), predict['y']-(predict["height"]//2), predict['width'], predict['height'], predict['class']
            end_x, end_y = (start_x + width), (start_y + height)
            print("image : ",dalle_image[start_y:end_y, start_x:end_x, :])
            yolo_result.append(dalle_image[start_y:end_y, start_x:end_x, :]) 
            result_cates.append(category)
            
    return yolo_result, result_cates


def extract_features(img,model):
    print("extract_features~~~~~~~~~")
    img = Image.fromarray(img).resize((384,384))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    print("features:",features)
    return features.flatten()

def search_similar_images(title):
    print("search_similar_images~~~~~~~~~~")
    result = []
    pred_img, pred_category = predict_yolo(title)
    dataset_paths = ds_create()
    index = faiss.read_index("static/index_efficientnet.faiss")
    for img in pred_img:
        query_features = extract_features(img,similar_model)
        print("chat_bot : ",query_features)
        distances, indices = index.search(np.array([query_features]),3)
        
        similar_images = [dataset_paths[i] for i in indices[0]]
        result += similar_images
    print("result",result)
    print("pred_category",pred_category)
    # return result, pred_category
    
    search_outfit(title)
    search_cates(pred_category)
    search_results(result)
    sequence_end("====================================================================================")
    
    return jsonify({"result": result, "pred_category": pred_category})