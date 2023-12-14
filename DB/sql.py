from config import mysql_config
import pymysql, hashlib
from datetime import date

HOST, PORT, DB, USER, PASSWORD = mysql_config()

# DB 연결 하기
def connection():
    try:
        con =  pymysql.connect(
                    host = HOST,
                    port = PORT,
                    db = DB,
                    user = USER,
                    password = PASSWORD
                )
        return con
    except pymysql.Error:
        print(pymysql.Error)


# 크롤링 함수
# ===============================================================================

# SQL 무신사 S3 저장된 이미지 가져오기
def musinsa_img_name(table):
    data = list()
    con = connection()
    cur = con.cursor()
    sql = f"SELECT img_path FROM {table} ORDER BY id"
    cur.execute(sql)
    for path in cur.fetchall():
        data.append(path[0])
    return data

# 중복 체크
def duplicate_check(img_path, category):
    if category == "Outer":
        table = "musinsa_outer"
    elif category == "Top":
        table = "musinsa_top"
    elif category == "Bottom":
        table = "musinsa_bottom"
    else:
        table = "musinsa_onepiece"
        
    conn = connection()
    cur = conn.cursor()
    sql = f"SELECT img_path FROM {table} WHERE img_path = '{img_path}'"
    try:
        cur.execute(sql)
        data = cur.fetchall()
    except pymysql.Error as e:
        print(e)
        data = None
    conn.close()
    return data

# 무신사 데이터 업로드
def musinsa_data_db_upload(table, img_path, price):
    conn = connection()
    cur = conn.cursor()
    sql = f"INSERT INTO {table} (img_path, price) VALUES (%s, %s)"
    cur.execute(sql, (img_path, price) )
    conn.commit()
    conn.close()
# ===============================================================================


# WEB Service
# ===============================================================================
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
# ===============================================================================


# Model
# ===============================================================================

# ===============================================================================



if __name__ == "__main__":
    a = musinsa_img_name("musinsa_onepiece")
    print(type(a), len(a))