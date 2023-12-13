# # 모듈 불러오기
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
# from tensorflow.keras.models import Model
# from roboflow import Roboflow
from PIL import Image
import openai
import json
import numpy as np
import faiss
import PIL
import os
import cv2
import csv
import urllib.request 
import requests
from io import BytesIO # 이미지 다운없이 바로 처리하기위해 
import ray 
import random
import os
import asyncio

from concurrent.futures import ThreadPoolExecutor


# 회원정보 목록  : 아이디 , 비번, 나이 , 성별 , 이름 , 이메일 , 폰

#ray.init()
def get_openai_key(): 
    key = None 
    try : 
        # 개발시 로컬 파일     
        key_path = 'openai_key.json'

        with open(key_path) as f :
            data = json.load(f)  # data = key 
            #print(data)
        key = data["OPENAI_API_KEY"] # 로컬에서 가져오기 
        
    except Exception : 
        # AWS Lambda 의 환경 변수 
        key = os.environ['OPENAI_API_KEY'] # 환경변수에 저장된 키 가져오기 . 
    return key
api_key = get_openai_key()
openai.api_key = api_key


# 달리 -------

client = openai.OpenAI(api_key = api_key)

def generate_image_sync(prompt, client):
    num = random.randrange(2, 11)
    num2 = random.randrange(2, 11)

    res_story = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size='1024x1024',
        quality='standard',
        n=1
    )
    save_path = f'{prompt.split()[-num]}_{num2}.png'
    url = res_story.data[0].url  # URL received from dalle
    urllib.request.urlretrieve(url,save_path)
    
    return save_path

async def dalle(prompt, client):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        save_path = await loop.run_in_executor(pool, generate_image_sync, prompt, client)
    return save_path


# # Example input:
# - Gender: Female
# - Seoson: winter
# - Situation: Travel
# - Preference: Street fashion

def current_season() :
    return "winter" 
     
user_data = {"age":"40" , 'gender' : 'male' , 'body_type' : 'chubby'}
age , gender , body_type = user_data.values()
season = current_season()
gpt_system_prompt = f"""
You're a chatbot that helps people choose what to wear today. According to the user's fashion taste, you need to help them choose the right outfit and print it out as an image. To make outfit recommendations, you need the following information.
< base conditions : don't ask > 
- nation: korean
- season : {season} 
- age : {age} 
- gender : {gender}
- body type : {body_type}
<required conditions>
- Situation (example : go on a date, go to a cafe, have an interview, go to work ...  )
- Clothing preferences ( example : streat fashion , casual fashion , dark or bright color)
To create a prompt to generate an image, the process is as follows:
1. Collect information from the user by asking the required conditions.
2. Once you have enough information, complete the prompt as shown below:
Example prompt:"Draw only 1 character,The character must be stand and include shoes and pants,realistic,8k uhd,soft lighting,high quality,20s Korean woman with a chubby body type,looking straight ahead, She is dressed in warm , the fashion is suitable for colder weather and going to work,The style should be cozy yet work-appropriate. "
3. pass the completed prompt as an argument to call the `dalle` function.
4. You must answer in Korean for user but final prompt you requried must be in English.
5. You don't have to explain about function or prompt to the user
6. The prompt should be no more than 400 characters long
"""

async def answer(state, state_chatbot, text):
    final_prompt = ""
    messages = state + [{
        "role": "user",
        "content": text
    }]

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=[{
            "name": "dalle",
            "description": "Generate an outfit image from prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt that generate an outfit image",
                    },
                },
                "required": ["prompt"],
            },
        }],
        function_call="auto",
    )

    print('content : ' ,completion.choices[0].message.content) # gpt 답변
    print('펑션 : ', completion.choices[0].message.function_call) # 펑션 활성화 여부 

    message_result = completion.choices[0].message.content if completion.choices[0].message.content else "" # 있으면 하고 없으면 말고

    new_state = [{
        "role": "user",
        "content": text
    }, {
        "role": "assistant",
        "content": message_result
    }]

    state = state + new_state
    function = completion.choices[0].message.function_call
    
    final_prompt = ""
    saved_three_image_url = None
    
    if function : # 펑션이 활성화 되면
        final_prompt += function.arguments.replace("\n", "").split(":")[1][:-1].replace("\"", "") # 프롬프트만 빼기 
        try:
            num = random.randrange(2, 11)
            num2 = random.randrange(2, 11)


            saved_three_image_url = await asyncio.gather(
                    
            dalle(final_prompt, client),
            dalle(final_prompt, client),
            dalle(final_prompt, client)
        )

            print("three_saved_Image_path : ", saved_three_image_url)
        except Exception as e:
            print("An error occurred:", e)

    else:
        state_chatbot = state_chatbot + [(text,message_result)]

    return state, state_chatbot, final_prompt , saved_three_image_url


# vscode 에서 확인하기 위함 ======   
state = ([{
        "role": "system",
        "content": gpt_system_prompt
    }])
state_chatbot = ([])
while True:
    prompt = input('ex 출근할때 입을 옷 추천해줘 -->> ')
    if prompt == 'z':
        break
    else:
      try :
        state, state_chatbot,final_prompt , saved_three_image_url = asyncio.run(answer(state, state_chatbot, prompt))
      except Exception as e :
        print("끝" , e )
        break
      finally :
        if saved_three_image_url : # 이미지 생성되면 종료
            print("finally " ,  final_prompt , "img_path : ",  saved_three_image_url )

            break
        

            
            