from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from DB.s3 import s3
from PIL import Image
import numpy as np
import faiss
import PIL
import cv2
from flask import jsonify
import datetime
import keras
from DB.sql import musinsa_img_name
from ultralytics import YOLO

today = datetime.date.today()
now = datetime.datetime.now()

search_log = f"config/log/search/{today}_search.log"


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

def yolo_model_create():
    print("Yolo model_create~~~~~~~~~~")
    trained_yolo = YOLO('engine/yolo_trained_model.pt')
    return trained_yolo



def similar_model_create():
    print("Similar model_create~~~~~~~~~~")
    detect_model = keras.models.load_model('engine/fine_tuning_model.h5')
    similar_model = Model(inputs=detect_model.input,outputs=detect_model.get_layer('global_average_pooling2d').output)
    return similar_model

trained_yolo = yolo_model_create()
similar_model = similar_model_create()

def predict_yolo(title):
    
    print("predict_yolo~~~~~~~~~~~")
    img_path = "web_service/static/images/created_image/" + title + ".png"
    
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
                start_x, start_y, category = int(x - (w / 2)) , int(y - (h / 2)) , int(cord[0].boxes.cls.cpu().numpy()[0])
                end_x, end_y = start_x + int(w), start_y + int(h)
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




def db_data_call(category):
    index = faiss.read_index(f"DB/S3_{category}_L2_index.faiss")
    s3_file_list = musinsa_img_name(f"musinsa_{category}")
    return index, s3_file_list



def search_similar_images(title):
    """
          카테고리 분류 : 'jacket' , 'shirt' , 'pants' ,' skirt' , 'dress'
                            = 0 ,       =1       =2        =3        =4 
    """
    category = {0 : "outer", 1 : "top", 2 : "bottom", 3 : "skirt", 4 : "onepiece"}
    
    print("search_similar_images~~~~~~~~~~")
    n_results = 3
    result = []
    pred_img, pred_category = predict_yolo(title)
    
    for img, category_idx in zip(pred_img, pred_category):
        
        index, s3_file_list = db_data_call(category[category_idx])
        query_features = extract_features(img, similar_model)
        distances, indices = index.search(np.array([query_features]), n_results)
        print("거리 : " , distances)
        similar_images = [s3_file_list[i].replace(".jpg", "") for i in indices[0]]
        result += similar_images
        
    
    search_outfit(title)
    search_cates(pred_category)
    search_results(result)
    sequence_end("====================================================================================")
    
    return jsonify({"result": result, "pred_category": pred_category})