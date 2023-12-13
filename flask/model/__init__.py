from openai import OpenAI
import openai
import os, json
import torch
from diffusers import DiffusionPipeline
import threading
import time
import queue as q
import urllib.request as req

# =============================================================================
# openai key 설정
# =============================================================================
def get_openai_key(): 
    key = None
    try:   
        # 개발시 로컬파일
        # openai_key.json 파일을 읽어서 "OPENAI_API_KEY" 키값 획득
        key_path = "C:\\Users\\user\\Desktop\\codiya\\flask\\openai_key.json"
        with open(key_path) as f:
            data = json.load(f)
            #print( data['OPENAI_API_KEY'][:5] )
        key = data['OPENAI_API_KEY']
    except Exception:
        # AWS Lambda의 환경변수
        key = os.environ['OPENAI_API_KEY']
    return key


# =============================================================================
# openai 객체 생성
# =============================================================================
client = OpenAI(api_key = get_openai_key())


# =============================================================================
# musinsa-igo 호출
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "youngmki/musinsaigo-2.0"
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32
)
pipe = pipe.to(device)
pipe.load_lora_weights(model_id)


# =============================================================================
# gpt_system_prompt 설정
# =============================================================================
gpt_system_prompt = """You're a chatbot that helps people choose what to wear today. Based on the user's mood and preferences, you need to help them choose the right outfit and print it out as an image. To make outfit recommendations, you need the following information.
- The user's gender
- Nationality
- Age
- Seoson
- Situation
- Clothing preferences

To create a prompt to generate an image, the process is as follows:
1. Collect information from the user by asking them a number of questions.
2. Once you have enough information, complete the prompt as shown below:
Example input:
- Gender: Female
- Nationality: Korea
- Age: 25
- Seoson: winter
- Situation: Travel
- Preference: Street fashion
Example prompt: In winter, a Korean woman in her 20s wearing a brown coat and pants for travel.
3. pass the completed prompt as an argument to call the `generate_image` function.
4. You should answer in Korean but, prompt must be in English.
"""

# =============================================================================
# generate_image 함수
# =============================================================================
def make_prompt(prompt: str) -> str:
    prompt_prefix = "RAW photo"
    prompt_suffix = "(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    return ", ".join([prompt_prefix, prompt, prompt_suffix]).strip()


def make_negative_prompt(negative_prompt: str) -> str:
    negative_prefix = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), \
    text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, \
    extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, \
    bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
    extra arms, extra legs, fused fingers, too many fingers, long neck"

    return (
        ", ".join([negative_prefix, negative_prompt]).strip()
        if len(negative_prompt) > 0
        else negative_prefix
    )

def generate_image(prompt):
    image = pipe(
        prompt=make_prompt(prompt),
        height=1024,
        width=768,
        num_inference_steps=15,
        guidance_scale=7.5,
        negative_prompt=make_negative_prompt(''),
        cross_attention_kwargs={"scale": 0.75},
    ).images[0]

    return image

# =============================================================================
# GPT 텍스트 QA
# =============================================================================
def answer(state, state_chatbot, text):
    messages = state + [{
        "role": "user",
        "content": text
    }]

    global client
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=[{
            "name": "generate_image",
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

    msg = completion.choices[0].message.content if completion.choices[0].message.content else ""

    new_state = [{
        "role": "user",
        "content": text
    }, {
        "role": "assistant",
        "content": msg
    }]

    state = state + new_state

    if completion.choices[0].message.function_call:
        function_name = completion.choices[0].message.function_call.name
        arguments = completion.choices[0].message.function_call.arguments
        arguments = json.loads(arguments)

        if function_name == "generate_image":
            print(arguments)

            state_chatbot = state_chatbot + [(text, f'{msg}')]
    else:
        state_chatbot = state_chatbot + [(text, msg)]

    return state, state_chatbot

def moderate_prompt(prompt: str) -> str:
    prompt_prefix = "Frotnt view, Full body shot"
    prompt_suffix = "High Noon, 16k uhd, ultra-realistic, soft lighting, film grain, Fujifilm XT3"

    negative_prefix = 'Do not draw as below'
    negative_suffix = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), \
    text, close up, cropped, out of frame, worst qualiy, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, \
    extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, \
    bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, \
    extra arms, extra legs, fused fingers, too many fingers, long neck"

    negative_pronmpt = "".join([negative_prefix + negative_suffix])

    return ", ".join([prompt.replace('.', ''), prompt_prefix, prompt_suffix, ]).strip() + "\n" + negative_pronmpt

final_prompt = moderate_prompt()

res_story = client.images.generate(
        model = 'dall-e-3',
        prompt = final_prompt,
        size = '1024x1792',
        quality = 'hd',
        n = 1
    )

res_story.url