from DB.sql import connection
from datetime import datetime, timedelta, date
from flask import Flask, render_template, request, current_app, jsonify, Response, redirect, url_for
from flask_socketio import SocketIO, emit
from models import search_similar_images
from openai import OpenAI
import jwt, hashlib
import config, time
from chatbot import answer
import asyncio



# 회원가입
def insert_userinfo_to_db(result:dict):
    # 비밀번호 해쉬 암호화
    print(result["password"])
    hash_security = hashlib.sha256()
    hash_security.update(result['password'].encode("utf-8"))
    password = hash_security.hexdigest()
    
    id = result['userid']
    name = result['username']
    birth = result['birthday']
    gender = result['gender']
    email = result['email']
    phone = result['phone']
    today = date.today()
    age = today.year - int(birth[:4])
    bodyshape = result['bodyshape']
    
    sql = "INSERT INTO member (id, password, name, birth, gender, email, phone, age, bodyshape) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

    conn = connection()
    cur = conn.cursor()
    try:
        cur.execute(sql, (id, password, name, birth, gender, email, phone, age, bodyshape))
    except pymysql.Error:
        print("중복된 아이디")

    conn.commit()
    conn.close()

# =====================================
# 정보 추출
def user_info(id, password):

    hash_security = hashlib.sha256()
    hash_security.update(password.encode("utf-8"))
    password = hash_security.hexdigest()

    conn = connection()
    cur = conn.cursor()
    cur.execute(f"SELECT gender, age, bodyshape FROM member WHERE id = '{id}' AND password = '{password}'")
    userdata = cur.fetchall()
    if userdata:   
        gender = userdata[0][0]
        age = userdata[0][1]
        bodyshape = userdata[0][2]
    conn.close()
    return gender, age, bodyshape
    
# 로그인 조회   
def login_db(id, password):
    login = False
    hash_security = hashlib.sha256()
    hash_security.update(password.encode("utf-8"))
    password = hash_security.hexdigest()

    conn = connection()
    cur = conn.cursor()

    cur.execute(f"SELECT id, password FROM member WHERE id = '{id}' AND password = '{password}'")
    userdata = cur.fetchall()
    if userdata:
        login = True
    conn.close()
    return login

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
    
    


