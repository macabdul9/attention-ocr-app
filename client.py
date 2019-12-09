import requests 

URL = "http://127.0.0.1:5000/post"


name = "abdul"
data = {"name":name}

r = requests.post(url = URL, data=data)

print(r.text)