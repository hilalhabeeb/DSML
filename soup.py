import requests
from bs4 import BeautifulSoup

def parser(url):
    response=requests.get(url)
    if response.status_code==200:
        soup=BeautifulSoup(response.content,'html.parser')
        print(soup.title.string)
        print(soup.get_text())
    else:
        print("error")

url="https://ajce.in"
parser(url)