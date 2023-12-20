import requests

def crawler(url):
    response =  requests.get(url)
    if response.status_code ==200:
        print(response.text)
    else:
        print("error")

url="https://ajce.in"
crawler(url)