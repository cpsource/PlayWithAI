import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pickle
import os
import copy

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

class CloneCompanyData:
    def __init__(self,other):
        self.info = copy.deepcopy(other.info)
        self.history = copy.deepcopy(other.history)
        self.history_metadata = copy.deepcopy(copy.deepcopy(other.history_metadata))
        self.actions = copy.deepcopy(other.actions)

        exit(0)
        
        self.dividends = copy.deepcopy(other.dividends)
        self.splits = copy.deepcopy(other.splits)
        self.capital_gains = copy.deepcopy(other.capital_gains)
        self.income_stmt = copy.deepcopy(other.income_stmt)
        self.quarterly_income_stmt = copy.deepcopy(other.quarterly_income_stmt)
        self.balance_sheet = copy.deepcopy(other.balance_sheet)
        self.quarterly_balance_sheet = copy.deepcopy(other.quarterly_balance_sheet)
        self.cashflow = copy.deepcopy(other.cashflow)
        self.quarterly_cashflow = copy.deepcopy(other.quarterly_cashflow)
        self.major_holders = copy.deepcopy(other.major_holders)
        self.institutional_holders = copy.deepcopy(other.institutional_holders)
        self.mutualfund_holders = copy.deepcopy(other.mutualfund_holders)
        self.earnings_dates = copy.deepcopy(other.earnings_dates)
        self.isin = copy.deepcopy(other.isin)
        self.options = copy.deepcopy(other.options)
        self.news = copy.deepcopy(other.news)
        self.option_chain = copy.deepcopy(other.option_chain)
        self.shares_full = copy.deepcopy(other.shares_full)
    
class CompanyData:
    
    def __init__(self):
        self.info = msft.info
        self.history = msft.history
        self.history_metadata = msft.history_metadata
        self.actions = msft.actions
        self.dividends = msft.dividends
        self.splits = msft.splits
        self.capital_gains = msft.capital_gains
        # show financials:
        # - income statement
        self.income_stmt = msft.income_stmt
        self.quarterly_income_stmt = msft.quarterly_income_stmt
        # - balance sheet
        self.balance_sheet = msft.balance_sheet
        self.quarterly_balance_sheet = msft.quarterly_balance_sheet
        # - cash flow statement
        self.cashflow = msft.cashflow
        self.quarterly_cashflow = msft.quarterly_cashflow
        # see `Ticker.get_income_stmt()` for more options
        # show holders
        self.major_holders = msft.major_holders
        self.institutional_holders = msft.institutional_holders
        self.mutualfund_holders = msft.mutualfund_holders
        # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
        # Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
        self.earnings_dates = msft.earnings_dates
        # show ISIN code - *experimental*
        # ISIN = International Securities Identification Number
        self.isin = msft.isin
        # show options expirations
        self.options = msft.options
        # show news
        self.news = msft.news
        # get option chain for specific expiration
        self.option_chain = msft.option_chain('2023-06-30')
        # show share count
        self.shares_full = msft.get_shares_full(start="2022-01-01", end=None)
        
        pass

file_path = 'wmt-CompanyData.pkl'

if os.path.exists(file_path):
    print("Loading from cache")
    with open(file_path,"rb") as f:
        clone_data = pickle.load(f)
else:
    print("Loading from Yahoo")
    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),)
    msft = yf.Ticker("WMT", session=session)    
    data = CompanyData()
    print(dir(data))
    #bucket_class.close()
    #data.backend.close()
    #exit(0)
    
    clone_data = CloneCompanyData(data)
    
    #clone_data = CloneCompanyData()
    # field by field copy
    #clone_data.info = data.info

    #clone_data = copy.deepcopy(data)
    
    with open(file_path,"wb") as f:
        pickle.dump(clone_data,f)
    
# msft must be setup before the class constructor    
#msft = yf.Ticker("WMT", session=session)    
#data = CompanyData()

#ticker = yf.Ticker("WMT", session=session)
#print(ticker.info)

print(clone_data.info)

#
# Now save onto disk
#

