from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO
import urllib.request
import time
import os
import requests
import random
import csv
from s3 import s3_upload_img
from sql import duplicate_check, musinsa_data_db_upload

start_url = "https://www.musinsa.com/categories/item/001005"



# Selenium Web 불러오기
def get_driver(url : str):
    chrome_options = Options()
    
    # 시크릿 모드
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()), options = chrome_options)
    driver.get(url)
    driver.implicitly_wait(10)
    return driver


# category_code 얻기
def get_category(url):
    # category 담을 버퍼
    category_code = {}

    # 드라이브 획득
    driver = get_driver(url)
    
    # 기달리기
    driver.implicitly_wait(10)

    for num in range(2, 7):
        driver.find_element(By.XPATH, f'//*[@id="leftCommonPc"]/div/section[1]/div[1]/nav/div[{num}]/div[1]/button').click()
        driver.implicitly_wait(2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        clothes_type = soup.select_one(f"#leftCommonPc > div > section.sc-g0sulw-0.bvYiGx > div:nth-child(2) > nav > div:nth-child({num}) > div.sc-8hpehb-1.YxBZW > button > div.sc-8hpehb-3.HnLvd > span").text
        code_data = soup.select(f"#leftCommonPc > div > section.sc-g0sulw-0.bvYiGx > div:nth-child(2) > nav > div:nth-child({num}) > div.sc-8hpehb-7.liOFHO > ul > li > a")
        codes = [code["href"][40:] for code in code_data[1:]]
        category_code[clothes_type] = codes
        time.sleep(2)
    driver.quit()
    return category_code

# category_code 상세 제품 url 획득
def get_detailed_product(code):
    # url_buffer
    url_buffer = list()

    # 후기순 & 제품 100
    for page in range(1, 4):
        url = f"https://www.musinsa.com/categories/item/{code}?d_cat_cd={code}&brand=&list_kind=small&sort=emt_high&sub_sort=&page={page}&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&plusDeliveryYn=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure="
        driver = get_driver(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.select("#searchList > li > div.li_inner > div.article_info > p.list_info> a")
        url_buffer += ["https:" + url["href"] for url in data]
        driver.quit()
        random_time = random.randint(1, 2)
        time.sleep(random_time)
    return url_buffer
        
# 이미지 데이터 저장 함수
def get_image_data(soup, url):
    image_name = f"{url[34:]}.jpg"

    img_url = f'https:{soup.select_one("#bigimg")["src"]}'
    
    response = requests.get(img_url)
    
    img_data = BytesIO(response.content)
    time.sleep(2)
    
    
    
    # 이미지 저장
    # urllib.request.urlretrieve(img_url, img_path)
    s3_upload_img(img_data, image_name)
    
    return image_name

# 데이터 저장 이후 몽고 디비 교체
def data_save(category, musinsa_price, review, img_path):
    if category == "Outer":
        table = "musinsa_outer"
    elif category == "Top":
        table = "musinsa_top"
    elif category == "Bottom":
        table = "musinsa_bottom"
    elif category == "Skirt":
        table = "musinsa_skirt"
    else:
        table = "musinsa_onepiece"
        
    # colum 데이터 구성
    data = {"price" : musinsa_price, "review" : review, "img_path" : img_path}
    
    # 파일 이름명
    file_name = f"{category}_data.csv"
    
    # 처음 생성 시 col까지 같이 생성
    try:
        with open(file_name, 'a', encoding = "utf8") as f:
            writer = csv.DictWriter(f, fieldnames = data.keys())
            writer.writerow(data)
    except:
        header = ["price", "review", "img_path"]
        if not os.path.exists(file_name):
            with open(file_name, 'w', encoding = "utf8") as f:
                writer = csv.DictWriter(f, fieldnames = header)
                writer.writeheader()
    musinsa_data_db_upload(table, img_path, musinsa_price)
    
# 크롤링 동작
def crawling(url, category):
    driver = get_driver(url)
    try:
        # 스타일 기준으로 클릭하기
        driver.find_element(By.XPATH, '//*[@id="estimateBox"]/div[2]/ul/li[1]').click()
        driver.implicitly_wait(10)
        
        # 유용한 순으로 정렬하기
        select = Select(driver.find_element(By.XPATH, '//*[@id="reviewSelectSort"]'))
        select.select_by_visible_text("유용한 순")
        driver.implicitly_wait(10)
    
    except:
        pass
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

        
    # 리뷰 데이터 획득
    try:
        reviews = soup.select("#reviewListFragment > div > div.review-contents > div.review-contents__text")
        
        # 문자열 변환
        review = "\n".join([review.text for review in reviews])
    except:
        review = None
        
    try:
    # 가격 데이터 획득
        musinsa_price = soup.select_one("#goods_price").text.strip()

    # 가격 int형 변환 
        musinsa_price = int(musinsa_price.strip().replace(",", "").replace("원", ""))
    except:
        musinsa_price = 0
    random_time = random.uniform(2, 5)
    time.sleep(random_time)
    
    data = duplicate_check(f"{url[34:]}.jpg", category)
    if not data:
    # 이미지 데이터 저장
        img_path = get_image_data(soup, url)
        data_save(category, musinsa_price, review, img_path)
        driver.quit()
    else:
        driver.quit()
    


def main(category):
    # 카테고리 전체 코드 획득
    category_code = get_category(start_url)
    #for category in category_code:
    codes = category_code[category]
    for code in codes:

        code_product_urls = get_detailed_product(code)
        print(f"category : {category}\ncode num : {code}\n============크롤링 시작============")
        for product_url in code_product_urls:
            crawling(product_url, category)
        print(f"category : {category}\ncode num : {code}\n============크롤링 종료============")
if __name__ == "__main__":
    category = "Top"
    main(category)


