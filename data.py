from hqth_alpha.config import *
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle as pkl
import os

#mysql_engine
def make_engine(db):
    engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8'%(tonglian_user,
            tonglian_passwd,tonglian_ip,tonglian_port,db),echo=False,encoding="utf-8")
    return engine

def tonglian_engine(): return make_engine(tonglian_db)
def quant_db_engine(): return make_engine(quant_db)
def factset_engine(): return make_engine(factset_db)

#load data from mysql
def get_trade_days(engine,begin_year='2000',end_year=None):
    if end_year is None:
        trade_calendar = pd.read_sql(f'select calendar date, is_open from md_trade_cal where exchange_cd = "XSHG" and \
        calendar_date >= "{begin_year}-01-01"',engine)
    else:
        trade_calendar = pd.read_sql(f'select calendar_date, is_open from md_trade_cal where exchange_cd = "XSHG" and \
        calendar_date >="{begin_year}-01-01" and calendar_date<="{end_year}-12-31"',engine)
    natural_days = trade_calendar['calendar_date'].tolist()
    trade_days = trade_calendar.loc[trade_calendar['is_open']==1,'calendar_date'].tolist()
    return trade_days,natural_days

def get_ohlc_data(engine,begin_date = '2010-01-01',end_date=None):
    if end_date is None:
        ohlc_data = pd.read_sql(f'select * from mkt_equd where TRADE_DATE>="{begin_date}" and \
        LEFT(TICKER_SYMBOL,1)!="2" and LEFT(TICKER_SYMBOL,1)!="9"',engine)
    else:
        ohlc_data = pd.read_sql(f'select * from mkt_equd where TRADE_DATE>="{begin_date}" and TRADE_DATE<={end_date}" and \
        LEFT(TICKER_SYMBOL,1)!="2" and LEFT(TICKER_SYMBOL,1)!="9"',engine)
    ohlc_dict = {}
    for col in ohlc_data.columns:
        if col not in ['ID','SECURITY_ID','TICKER_SYMBOL','EXCHANGE_CD']:
            ohlc_dict[col] = ohlc_data.pivot(index="TRADE_DATE",columns='TICKER_SYMBOL',values=col)\
                .sort_index().sort_index(axis=1)
    return ohlc_dict
