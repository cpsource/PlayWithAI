import sys
sys.path.insert(0, "./client-python")
from polygon import RESTClient
import capikey
#client = RESTClient(api_key="<API_KEY>")
client = capikey.client

#client = RESTClient()

print(client)

ticker = "AAPL"

if False:
    # List Aggregates (Bars)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_="2023-01-01", to="2023-06-13", limit=50000):
        aggs.append(a)
    print(aggs)

if True:
    # Get Last Trade
    trade = client.get_last_trade(ticker=ticker)
    print(trade)

if False:
    # List Trades
    trades = client.list_trades(ticker=ticker, timestamp="2023-01-04")
    for trade in trades:
        print(trade)

if False:
    # Get Last Quote
    quote = client.get_last_quote(ticker=ticker)
    print(quote)

if False:
    # List Quotes
    quotes = client.list_quotes(ticker=ticker, timestamp="2022-01-04")
    for quote in quotes:
        print(quote)
