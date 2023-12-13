# from sklearn.metrics.pairwise import cosine_similarity
# import h5py # 모델이 추출한 무신사 데이터셋의 특징 저장
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import models
from tensorflow.keras.models import Model
import numpy as np
import faiss
import PIL
import os
import io
import cv2
from sql import musinsa_img_name
from s3 import s3_download_img


# 모델 생성
def model_create():
    # EfficientNetV2S 모델 객체 생성
    base_model = models.load_model("/Users/mingi/Downloads/fine-tuning-model/fine_tuning_model.h5")
    # EfficientNetV2S 모델의 데이터 추출 모드 사용 
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    return model

def extract_features(img_path, model):
    response = s3_download_img(img_path)
    img_data = response["Body"].read()
    
    img = image.load_img(io.BytesIO(img_data), target_size=(512, 512))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# 벡터디비 생성
def db_create(model):
    # Initialize FAISS index
    dimension = model.output.shape[1]  # This should match the feature vector size
    index = faiss.IndexFlatL2(dimension)
    return index

# Now you can use the 'index' for similarity searches

## ----------- 이미지 서치하는 함수 
def search_similar_images(query_img_path, model, index, dataset_paths, n_results=5):
    """
    Search for similar images in the FAISS index.

    :param query_img_path: Path to the query image.
    :param model: Pretrained Keras model for feature extraction.
    :param index: FAISS index containing features of the dataset.
    :param dataset_paths: List of paths to images in the dataset.
    :param n_results: Number of similar images to retrieve.
    :return: List of paths to similar images.
    """
    # Extract features from the query image
    query_features = extract_features(query_img_path, model)

    # Search the FAISS index
    distances, indices = index.search(np.array([query_features]), n_results)
    
    # Retrieve the similar images
    similar_images = [dataset_paths[i] for i in indices[0]]

    return similar_images

# DB table name
tables = ["musinsa_outer", "musinsa_top", "musinsa_bottom", "musinsa_onepiece"]

# 모델 생성
model = model_create()

# 이미지 이름 가져옴
for table in tables:
    table_img_names = musinsa_img_name(table)
    index = db_create(model)
    for img_name in table_img_names:
        features = extract_features(img_name, model)
        index.add(np.array([features]))

    faiss.write_index(index, f"{table}_index.faiss")