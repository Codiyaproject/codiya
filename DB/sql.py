import sys
sys.path.append("..")
from config import mysql_config
import pymysql

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
    sql = f"SELECT img_path FROM {table} WHERE = '{img_path}'"
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

if __name__ == "__main__":
    pass