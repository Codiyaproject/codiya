# 모듈 불러오기
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
import requests
#from io import BytesIO # 이미지 다운없이 바로 처리하기위해 
import random
''' 기본 프롬프트 
    "A realistic image of a single 나이 : {20-something} Korean woman with a {chubby} body type, 
    standing in full body with shoes, looking straight ahead. She is dressed in warm
    her overall style, and her posture."
'''
## dataset_names 획득 
dataset_names = list()
with open("dataset_names.csv",'r') as f :  

    file = csv.reader(f)
    for line in file:
        dataset_names+=line

print(len(dataset_names)) # 확인 41537 

# 모든 무신사 이미지 데이터 폴더 
img_dir = os.path.join(os.getcwd(), "data_storage/image")

categories = ['onepiece', 'outer','pants','skirt','top']
# 서버에 저장된 이미지 파일 이름 불러오기
def get_file_paths():

    # 모든 파일을 저장할 버퍼
    file_names = []   
    # filename만 모두 담기 root , dir 안씀  
    for root, _ , filename in os.walk(img_dir):
    
        if root.split("\\")[-1] in categories :  # 만약 현재 돌고 있는 root 가 카테고리 안에 있다면 (아니면 공백이나 다른거 들어옴)
            file_names += filename # os.walk 가 filename을 [] 형태로 뱉고있어서 append 말고 + 
    
    return file_names        

# 이미지 버퍼에 담기
#dataset_names = get_file_paths() # csv 파일로 대체 

# 패션이미지 category 분류 모델
def predict_yolo(rf_model, img_path, confidenct=40, overlap = 30):
    '''
    return : x, y 값 좌표 ( 예측된 사각형의 젤 왼쪽 위 끝점 임 )
            width , height
            class : 카테고리 이름
    '''
    # 이미지 불러오기 ( 나중에 자르기 위해서  )
    image = cv2.imread(img_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Yolo 모델로 이미지 예측하기
    predict = rf_model.predict(img_path, confidence = confidenct, overlap = overlap)
    #predict.save("yolo_test.jpg")
    # 예측 정보 출력
    predict_info = predict.json()
    # 잘 탐지했나 저장해서 테스트
    #predict.save("save_test.jpeg") 
    # 예측 이미지 저장 공간 생성
    yolo_result = list()
    # 이미지 카테고리 . # 
    result_categories = list()  # 이미지 인덱스 = 카테고리 인덱스 

    # 이미지 예측된 부위 데이터 담기
    for info in predict_info["predictions"]:
        if info["class"] not in ["hat", "sunglass", "bag", "shoe"]:
            start_x, start_y, width, height, category = info['x']-(info["width"] // 2), info['y']-(info["height"] // 2), info['width'], info['height'], info['class']
            end_x, end_y = (start_x + width), (start_y + height)
            #print(image[start_y:end_y, start_x:end_x, :].dtype)
            yolo_result.append(image[start_y:end_y, start_x:end_x, :]) 
            result_categories.append(category)
    num = random.randrange(2, 11)
    # 잘 잘렸나 저장해서 테스트
    #print("yolo_result" , len(yolo_result)) 
    #print(yolo_result) 

    # for i in yolo_result :  
               
    #      im = Image.fromarray(i)        
    #      im.save(f"yolotest_{num}.jpeg")     
    return yolo_result, result_categories

# 모델 생성하기 
def model_create():
    # EfficientNetV2S 모델 객체 생성
    base_model = EfficientNetV2S(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    
    # EfficientNetV2S 모델의 데이터 추출 모드 사용 -> 이미지 특성 모델
    similar_model = Model(inputs=base_model.input, outputs=base_model.get_layer('top_conv').output)
    
    # Roboflow 모델 생성 -> 이미지 식별 모델
    rf = Roboflow(api_key="Q78HnDQgOoukAA6rXrsG")
    project = rf.workspace().project("fashion-hkjfr")
    rf_model = project.version(5).model
    return similar_model, rf_model

# 이미지 크기 조정(input data) 만들기
def extract_features(img, model):
    #img_path = "/Users/mingi/Desktop/Sesac_Project/Fashion_Project/data_storage/images/" + img_path
    # img = image.load_img(img_path, target_size=(384, 384))
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    img = Image.fromarray(img).resize((384, 384))
    #img.save("extract_features_test.jpeg") 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# 생성된 이미지 category 분류 하기 유사 이미지 추천해주기
def search_similar_images(query_img_path, similar_model, rf_model, index, dataset_paths, n_results=3):
    """
    Search for similar images in the FAISS index.

    :param query_img_path: Path to the query image.
    :param model: Pretrained Keras model for feature extraction.
    :param index: FAISS index containing features of the dataset.
    :param dataset_paths: List of paths to images in the dataset.
    :param n_results: Number of similar images to retrieve.
    :return: List of paths to similar images.
    """
    result = []
    # Extract features from the query image
    pred_img, pred_category  = predict_yolo(rf_model, query_img_path)
    for img in pred_img:
        query_features = extract_features(img, similar_model)

        # Search the FAISS index
        distances, indices = index.search(np.array([query_features]), n_results)
        # Retrieve the similar images
        similar_images = [dataset_paths[i] for i in indices[0]]
        result += similar_images
    return result , pred_category

# 벡터 디비 호출
index = faiss.read_index("index_efficientnet.faiss")

# 모델 생성
similar_model, rf_model = model_create()
        
# 테스트 파일 
# image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-isiflj0QPpZOqxTrr3DNR0q8/user-LT48mRZarJXcn0PeAjmcuwrc/img-GkhndvGEPuwVKSqBACRGNrKD.png?st=2023-11-28T03%3A34%3A00Z&se=2023-11-28T05%3A34%3A00Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-28T02%3A51%3A55Z&ske=2023-11-29T02%3A51%3A55Z&sks=b&skv=2021-08-06&sig=GrXt2dc0esy8bFLD7%2BBM2g82nO8X2dLGIPncSKw1TsA%3D"

# res = requests.get(image_url)
# img = Image.open(BytesIO(res.content))
# img
query_img_path = 'data_storage\\target_data\\top\\wedding,_3.png'
similar_images,pred_category = search_similar_images(query_img_path, similar_model,rf_model, index, dataset_names, n_results = 3) 
print("상품 이름" , similar_images) 
print("카테고리 : ", pred_category)
# 유사한 이미지 출력
for img_path in similar_images:
    for category in ['/onepiece/', '/outer/','/pants/','/skirt/','/top/']: 
        final_path = img_dir + category + img_path        
        if os.path.exists(final_path):
            print(final_path)
            try:
                image = cv2.imread(final_path)
                cv2.imshow(f"{category}", image)
                cv2.waitKey(0)
            except Exception as e:
                print(e)
            finally:
                cv2.destroyAllWindows()

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
            
            