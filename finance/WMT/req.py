import requests

url = "https://stock-and-options-trading-data-provider.p.rapidapi.com/options/aapl"

headers = {
	"X-RapidAPI-Proxy-Secret": "a755b180-f5a9-11e9-9f69-7bf51e845926",
	"X-RapidAPI-Key": "SIGN-UP-FOR-KEY",
	"X-RapidAPI-Host": "stock-and-options-trading-data-provider.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())
