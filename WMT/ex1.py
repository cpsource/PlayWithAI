# See also: https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html

import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pickle
import os
#import copy
import chatgpt_moonphase as cmp
import numpy as np
import sqlite3

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

    def show(self):
        print(f"self.info = {self.info}\n")
        print(f"self.history = {self.history}\n")
        print(f"self.history_metadata = {self.history_metadata}\n")
        print(f"self.actions = {self.actions}\n")
        print(f"self.dividends = {self.dividends}\n")
        print(f"self.splits = {self.splits}\n")
        print(f"self.capital_gains = {self.capital_gains}\n")
        print(f"self.income_stmt = {self.income_stmt}\n")
        print(f"self.quarterly_income_stmt = {self.quarterly_income_stmt}\n")
        print(f"self.balance_sheet = {self.balance_sheet}\n")
        print(f"self.quarterly_balance_sheet = {self.quarterly_balance_sheet}\n")
        print(f"self.cashflow = {self.cashflow}\n")
        print(f"self.quarterly_cashflow = {self.quarterly_cashflow}\n")
        print(f"self.major_holders = {self.major_holders}\n")
        print(f"self.institutional_holders = {self.institutional_holders}\n")
        print(f"self.mutualfund_holders = {self.mutualfund_holders}\n")
        print(f"self.earnings_dates = {self.earnings_dates}\n")
        print(f"self.isin = {self.isin}\n")
        print(f"self.options = {self.options}\n")
        print(f"self.news = {self.news}\n")
        print(f"self.option_chain = {self.option_chain}\n")
        print(f"self.shares_full = {self.shares_full}\n")
        
# Ug, pickle won't save/restore with cache

file_path = 'wmt-CompanyData.pkl'
ticker = 'WMT'

if False and os.path.exists(file_path):
    print("Loading from cache")
    with open(file_path,"rb") as f:
        clone_data = pickle.load(f)
else:
    print("Loading from Yahoo")
    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),)
    msft = yf.Ticker(ticker, session=session)    
    data = CompanyData()
    data.show()

    print("History")
    
    d = msft.history(start="2023-06-28",end="2023-06-30",interval='1m')
    print(type(d),d)
    print(d.shape)
    print(d.index.shape)
    print(type(d.index[0]))
    max = len(d.index)
    print(max)
    print(d.index[0])
    tim = d.index[0].strftime('%Y-%m-%d %H:%M:%S')
    print(f"tim = {tim}\n")

    phase, phase_name = cmp.get_moon_phase(tim)
    print(f"phase = {phase}, name = {phase_name}\n")

    # check for start of second date
    tim1 = d.index[390].strftime('%Y-%m-%d %H:%M:%S')
    phase1, phase_name1 = cmp.get_moon_phase(tim1)
    
    print(f"tim1[390] = {tim1}\n")
    
    d1 = d[:390]
    X1 = d1['Close']
    d2 = d[390:390*2]
    X2 = d2['Close']

    print(type(X1),X1)
    print(X1.shape)

    # create numpy arrays
    np_X1 = np.array(X1, dtype=np.float32)
    np_X2 = np.array(X2, dtype=np.float32)

    # debug info
    print(type(np_X1),np_X1)
    print(np_X1.shape)

    # now convert np_Xn to blob with pickle
    # Serialize the tensor to a byte array
    np_X1_blob = pickle.dumps(np_X1)
    np_X2_blob = pickle.dumps(np_X2)

    # store in database

    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    try:
        # Insert the np array into the database 
        cursor.execute("INSERT INTO my_table (datetime_column, ticker, moon_phase, y1, y2, y3, closes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (tim,
                        ticker,
                        phase,
                        0,0,0,
                        sqlite3.Binary(np_X1_blob)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error occurred at {tim} while inserting row:", e)

    try:        
        # Insert the np array into the database 
        cursor.execute("INSERT INTO my_table (datetime_column, ticker, moon_phase, y1, y2, y3, closes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (tim1,
                        ticker,
                        phase1,
                        0,0,0,
                        sqlite3.Binary(np_X2_blob)))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error occurred at {tim1} while inserting row:", e)
    
    # Close the database connection
    conn.close()

    #for idx in range(max):
    #print(idx,d.index[idx])
        
    #bucket_class.close()
    #data.backend.close()
    #exit(0)
    
    #clone_data = CloneCompanyData(data)
    
    #clone_data = CloneCompanyData()
    # field by field copy
    #clone_data.info = data.info

    #clone_data = copy.deepcopy(data)
    
    #with open(file_path,"wb") as f:
    #pickle.dump(clone_data,f)
    
# msft must be setup before the class constructor    
#msft = yf.Ticker("WMT", session=session)    
#data = CompanyData()

#ticker = yf.Ticker("WMT", session=session)
#print(ticker.info)

#print(clone_data.info)

#
# Now save onto disk
#

