import requests

url = 'http://localhost:9696/predict'

response = requests.post(url, data="")
print(response._content.decode("utf-8"))