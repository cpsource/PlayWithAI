# Must handle closing early days such as
# jul3, tgiving eve, christmas eve

import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pickle
import os
import datetime
import chatgpt_moonphase as cmp
import numpy as np
import sqlite3
import present
import add_one_day as aod
import trading_day

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

stock_watch_list = ( "TJX",
	"AAPL",
        "MSFT",
        "AMZN",
        "BRK.A",
        "JNJ",
        "UNH",
        "WMT",
        "V",
        "MA",
        "HD",
        "GOOGL",
        "TSLA",
        "NVDA",
        "CRM",
        "MCD",
        "PFE",
        "ADBE",
        "COST",
        "KO",
        "ABBV",
        "PEP",
        "CVX",
        "MRK",
        "PG",
        "LLY",
        "AVGO",
        "XOM",
        "JPM",
        "BRK-B",
        "META",
        "GOOG" )

print("Loading from Yahoo")
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),)

def do_ticker(conn, ticker, start_date, end_date):
    # don't do if already done
    if present.check_record(conn,start_date,ticker):
        print(f"Ticker {ticker} already loaded")
        return
    msft = yf.Ticker(ticker, session=session)    
    d = msft.history(start=start_date, end=end_date,interval='1m')
    max = len(d.index)
    if max == 0:
        print(f"Ticker {ticker} no records loaded")
        return
    #print(f"max = {max}")
    #print(d.index[0])
    tim = d.index[0].strftime('%Y-%m-%d %H:%M:%S')
    #print(f"tim = {tim}\n")

    phase, phase_name = cmp.get_moon_phase(tim)
    #print(f"phase = {phase}, name = {phase_name}\n")

    # check for start of second date
    #tim1 = d.index[390].strftime('%Y-%m-%d %H:%M:%S')
    #phase1, phase_name1 = cmp.get_moon_phase(tim1)
    
    #print(f"tim1[390] = {tim1}\n")
    
    #d1 = d[:390]
    d1 = d
    X1 = d1['Close']
    #d2 = d[390:390*2]
    #X2 = d2['Close']

    # create numpy arrays
    np_X1 = np.array(X1, dtype=np.float32)
    #np_X2 = np.array(X2, dtype=np.float32)

    # now convert np_Xn to blob with pickle
    # Serialize the tensor to a byte array
    np_X1_blob = pickle.dumps(np_X1)
    #np_X2_blob = pickle.dumps(np_X2)

    # store in database

    # Connect to the SQLite database
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

    if False:
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
    
# Open the SQLite database
conn = sqlite3.connect('database.db')

#start_date_str = "2023-06-23"
#end_date_str   = "2023-07-01"
start_date_str = "2023-07-01"
end_date_str   = "2023-07-07"

for ticker in stock_watch_list:
    print(ticker)

    # do one day at a time
    cur_date_str = start_date_str
    end_d_tmp = datetime.datetime.strptime(cur_date_str, "%Y-%m-%d")
    end_d_tmp = aod.add_one_day(end_d_tmp)
    end_date_tmp_str = f"{end_d_tmp:%Y-%m-%d}"
        
    while not aod.compare_dates(end_date_tmp_str,end_date_str):

        if trading_day.is_wall_street_trading_day_str(cur_date_str):
            print(f"calling do_ticker with {cur_date_str}, {end_date_tmp_str}")
            do_ticker(conn,ticker,cur_date_str, end_date_tmp_str)
        else:
            print(f"Not a trading day: {cur_date_str}")
            
        # on to next date
        cur_d = datetime.datetime.strptime(cur_date_str, "%Y-%m-%d")
        cur_d = aod.add_one_day(cur_d)
        cur_date_str = f"{cur_d:%Y-%m-%d}"
        end_d_tmp = aod.add_one_day(end_d_tmp)
        end_date_tmp_str = f"{end_d_tmp:%Y-%m-%d}"
        
# Close the database connection
conn.close()
