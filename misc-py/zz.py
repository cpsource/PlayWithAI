import requests

response = requests.get("https://www.foobar.com")

if response.status_code == 200:
  print("The URL exists.")
else:
  print("The URL does not exist.")

