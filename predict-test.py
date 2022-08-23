import requests

url = 'http://localhost:9696/predict'

print()
data = input("Enter your text: ")
print()

response = requests.post(url, data=data)
print(response._content.decode("utf-8"))