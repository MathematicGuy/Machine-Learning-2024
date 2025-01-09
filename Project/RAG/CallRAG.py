import requests

url = 'http://127.0.0.1:5000.ngrok.io/generate'  # Replace with your ngrok URL
data = {'query': 'What are the symptoms of pellagra?'}

response = requests.post(url, json=data)
print(response.json())