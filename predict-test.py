import requests

url = 'http://localhost:9696/predict'

response = requests.post(url, json="").json()
print(response)