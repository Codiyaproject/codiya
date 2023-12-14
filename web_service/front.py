from DB.sql import insert_userinfo_to_db, user_info, login_db
from datetime import datetime, timedelta, date
from flask import Flask, render_template, request, current_app, jsonify, Response, redirect, url_for
from flask_socketio import SocketIO, emit
from engine.models import search_similar_images
from openai import OpenAI
import jwt, hashlib
import config, time
from engine.chatbot import answer
import asyncio


app = Flask(__name__)

# 보안키 설정
app.config.from_object(config)

@app.route('/', methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        
        user_id = request.form.get('uid')
        password = request.form.get('upw')
        login = login_db(user_id, password)
        if login:
            gender, age, bodyshape = user_info(user_id, password)
            payload = {
                "gender" : gender,
                "age" : age,
                "bodyshape" : bodyshape,
                # 시간 5시간 지속
                "exp" : datetime.utcnow() + timedelta(seconds = 60 * 60 * 5)
            }
            
            # 토큰 발급 -> 시크릿키, 해시알고리즘, 데이터
            SECRET_KEY = config.config
            # 토큰 발급
            token = jwt.encode(payload, SECRET_KEY, algorithm = "HS256")
            
            return jsonify({"code" : 1, "token" : token})
        else:
            print('잘못된 ID PW입니다')
    return render_template('public/login-page.html')

@app.route('/chatbot')
def chatbot():
    # 1. 쿠키중에 토큰 획득 -> 실패 -> 401
    token = request.cookies.get('token')
    SECRET_KEY = config.config # 환경변수
    print(token, SECRET_KEY)
    if not token or not SECRET_KEY:
        return Response(status=401)
    try:
        # 2. 디코딩 -> 실패하면 -> 401
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        # 3. 유효 날짜 추출 , 현제 시간 기준보다 과거인지 체크 => 과거라면:만료 -> 401        
        if payload['exp'] < time.mktime( datetime.utcnow().timetuple() ): # 현재시간보다는 과거
            return Response(status=401)
        # 4. 정상
        print(token, SECRET_KEY)
        # main(text)
        return render_template('pages/chatbot.html')
    except jwt.InvalidTokenError:
        return Response(status=401)
    except jwt.ExpiredSignatureError:
        return Response(status=401)
    except jwt.exceptions.DecodeError:
        return Response(status=401)    
    
@app.route('/nlp/signup')
def signup():
    return render_template('pages/sign-up.html')    

@app.route("/models/similar", methods=["POST"])
def get_similar():
    title = request.form['title']
    print("similar 왔어요~",title)
    return search_similar_images(title)

@app.route("/openai/text", methods=["POST"])
def chat_gpt():
    token = request.cookies.get('token')
    print(token)
    SECRET_KEY = config.config # 환경변수
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    print(payload)
    text = request.form['text']
    age = payload["age"]
    gender = payload["gender"]
    bodyshape = payload["bodyshape"]
    return asyncio.run(answer(text, age, gender,bodyshape))
    
@app.route('/model', methods = ['GET', 'POST'])
def model():
    if request.method == 'POST':
        result = request.form
        if result["password"] != result["confirm_password"]:
            pass
        else:
            insert_userinfo_to_db(result)
            
    return render_template('public/login-page.html')


app.run(host = "0.0.0.0", debug = True)
    
    


