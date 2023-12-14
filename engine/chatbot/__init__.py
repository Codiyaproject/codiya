from flask import jsonify
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from config import openai_config
import urllib.request
import threading
import requests
import datetime
import asyncio

today = datetime.date.today()
now = datetime.datetime.now()


dalle_log = f"/web_service/static/log/dalle/{today}_search.log"

def save_prompt(message):
    try:
        with open(dalle_log,'a') as log_file:
            log_file.write(f"{now}_Final_Prompt : {message}" + '\n')
    except FileNotFoundError:
        with open(dalle_log, 'w') as log_file:
            log_file.write(f"{now}_Final_Prompt : {message}" + '\n')
            
            
def save_image(message):
    try:
        with open(dalle_log,'a') as log_file:
            log_file.write(f"{now}_Created_Image : {message}" + '\n')
    except FileNotFoundError:
        with open(dalle_log,'w') as log_file:
            log_file.write(f"{now}_Created_Image : {message}" + '\n')     
            
            
def sequence_end(message):
    try:
        with open(dalle_log,'a') as log_file:
            log_file.write(f"{message}" + '\n')
    except FileNotFoundError:
        with open(dalle_log,'w') as log_file:
            log_file.write(f"{message}" + '\n')

lock = threading.Lock()



client = OpenAI(api_key = openai_config())

def moderate_prompt(prompt: str) -> str:
    prompt_prefix = "Front view, Full body shot"
    prompt_suffix = "High Noon, 16k uhd, ultra-realistic, soft lighting, film grain, Fujifilm XT3"

    negative_prefix = 'Do not draw as below'
    negative_suffix = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), \
        text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, \
        extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, \
        bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
        extra arms, extra legs, fused fingers, too many fingers, long neck"

    negative_prompt = "".join([negative_prefix + prompt + negative_suffix])

    return ", ".join([prompt.replace('.', ''), prompt_prefix, prompt_suffix, ]).strip() + "\n" + negative_prompt


def generate_image_sync(prompt):
    print('call create img')
    local_client = OpenAI(api_key = openai_config())
    get_image = local_client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size='1024x1024',
        quality='standard',
        n=1,
    )

    img = get_image.data[0].url
    title = img.split('/img-')[1].split('.png')[0]
    save_path = 'static/images/created_image/'+title+'.png'
    print(img)
    urllib.request.urlretrieve(img, save_path)
    return title

# 달리 병렬
async def dalle(prompt):
  
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        save_path = await loop.run_in_executor(pool,generate_image_sync,prompt)
        
    return save_path

def current_season():
    season = "Autumn"
    
    today = datetime.date.today()
    m = today.month
    if m in [12,1,2]:
        season = "Winter"
    elif m in [3,4,5]:
        season = "Spring"
    elif m in [6,7,8]:
        season = "Summer"
        
    print(m,season)
    
    return season

user_data = {"age":"40" , 'gender' : 'male' , 'bodyshape' : 'slim'}
age , gender , bodyshape = user_data.values()
season = current_season()
gpt_system_prompt = f"""
You're a chatbot that helps people choose what to wear today. According to the user's fashion taste, you need to help them choose the right outfit and print it out as an image. To make outfit recommendations, you need the following information.
< base conditions : don't ask > 
- nation: korean
- season : {season} 
- age : {age} 
- gender : {gender}
- body type : {bodyshape}
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

state = ([{
    "role": "system",
    "content": gpt_system_prompt
}])

state_chatbot = ([])

async def answer(text, age, gender, bodyshape):
       
    global state
    global state_chatbot
    print("chatbot", text)
    
    gpt_system_prompt = f"""You're a chatbot that helps people choose what to wear today. Based on the user's mood and preferences, you need to help them choose the right outfit and print it out as an image. To make outfit recommendations, you need the following information.
    < base conditions : don't ask > 
    - nation: korean
    - season : {season}
    - age : {age} 
    - gender : {gender}
    - body type : {bodyshape}
    <required conditions>
    - Situation (example : go on a date, go to a cafe, have an interview, go to work ...  )
    - Clothing preferences ( example : streat fashion , casual fashion , dark or bright color)
    To create a prompt to generate an image, the process is as follows:
    1. Collect information from the user by asking the required conditions.
    2. Once you have enough information, complete the prompt as shown below:
    Example prompt:"alone,solo,Draw only 1 character,The character must be stand and include shoes and pants,hyper realistic,8k uhd,soft lighting,high quality,20s Korean woman with a chubby body type,looking straight ahead, She is dressed in warm , the fashion is suitable for colder weather and going to work,The style should be cozy yet work-appropriate. "
    3. pass the completed prompt as an argument to call the `dalle` function.
    4. You must answer in Korean for user but final prompt you requried must be in English.
    5. You don't have to explain about function or prompt to the user
    6. The prompt should be no more than 400 characters long
    """
    
    state[0]['content'] = gpt_system_prompt

    messages = state + [
        {"role": "user", "content": text}
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
        max_tokens=2048,
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
        function_call="auto"

    )

    message_result = completion.choices[0].message.content if completion.choices[0].message.content else ""

    new_state = [{
        "role": "user",
        "content": text
    }, {
        "role": "assistant",
        "content": message_result
    }]

    state = state + new_state
    final_prompt = ""
    saved_three_image_url = None
    function = completion.choices[0].message.function_call
    
    if function:
        final_prompt += function.arguments.replace("\n", "").split(":")[1][:-1].replace("\"", "")
        state = ([{
            "role": "system",
            "content": gpt_system_prompt
        }])
        state_chatbot = ([])
        print(final_prompt)
        # final_prompt = moderate_prompt(final_prompt)
        print("final : ", final_prompt)
        
        try:
            saved_three_image_url = await asyncio.gather(
                dalle(final_prompt),
                dalle(final_prompt),
                dalle(final_prompt)
            )
            
        except Exception as e:
            print("An error occurred:",e)
        
        message_result = ""
        message_result += "created"
        final_img = saved_three_image_url
        
        save_prompt(final_prompt)
        save_image(final_img[0])
        save_image(final_img[1])
        save_image(final_img[2])
        sequence_end("====================================================================================")

    else:
        state_chatbot = state_chatbot + [(text, message_result)]
        final_img = []

    print("result", message_result)
    print("state", state)
    print("state_chatbot", state_chatbot)

    return jsonify({"result": message_result, "state": state, "state_chatbot": state_chatbot,"final_img":final_img})