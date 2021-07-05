from hqth_alpha.config import *
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime, date
from statmodels.api import OLS


# 处理数据列名格式为string
def format_columns(factor):
    try:
        c0 = factor.columns[0]
        if type(c0) == str:
            if c0[0] == 'S' or c0[0] == 's':
                factor.columns = [x[2:] for x in factor.columns]
            else:
                factor.columns = [format(int(x[:6]), '06d') for x in factor.columns]
        else:
            factor.columns = [format(int(x), '06d') for x in factor.columns]
    except:
        print("因子columns格式有误，请调整为'000001'字符串样式")
    return factor


# 处理数据index格式为datetime.date
def format_date_index(factor):
    try:
        i0 = factor.index[0]
        if type(i0) == str:
            try:
                factor.index = [datetime.strptime(str(int(x[:8])), "%Y%m%d").date() for x in factor.index]
            except:
                try:
                    factor.index = [datetime.strptime(x[:10].strip(), "%Y-%m-%d").date() for x in factor.index]
                except:
                    try:
                        factor.index = [datetime.strptime(x[:10].strip(), '%Y %m %d').date() for x in factor.index]
                    except:
                        print("因子index格式有误，请调整为datetime.date格式")
        elif type(i0) == datetime:
            factor.index = [x.date() for x in factor.index]
        elif type(i0) == pd.Timestamp:
            factor.index = [x.topydatetime().date() for x in factor.index]
        elif type(i0) == date:
            pass
        else:
            try:
                factor.index = [datetime.strptime(str(int(x)), '%Y%m%d').date() for x in factor.index]
            except:
                print("因子index格式有误,请调整为datetime.date格式，现为", type(i0), i0)
    except:
        print("因子index格式有误,请调整为datetime.date格式")
    return factor


def format_factor(factor):
    "将因子格式调整一致"
    return format_date_index(format_columns(factor))


# 对行业做中性化
def neutralize_industry(factor, industry):
    print("对因子进行行业中性化处理")
    new_factor = factor.apply(lambda x: x.groupby(industry.loc[x.name]). \
                              apply(lambda y: (y - y.mean()) / y.std()), axis=1)
    return new_factor


# 分解
def factor_decompose(factor, *args):
    ex_df = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for row in factor.iterrows():
        y = row[1]
        x = pd.DataFrame(index=y.index)
        for f_ in args:
            if type(f_) == list:
                for f in f_:
                    x = pd.concat([x, f.loc[row[0]]], axis=1, join='inner')
            else:
                x = pd.concat([x, f_.loc[row[0]]], axis=1, join='inner')

        # 保证x的行数与y的行数对齐
        x = x.reindex(y.index)
        x = x.fillna(x.mean())
        # 回归，求残差
        ex_val = OLS(y, x, missing='drop').fit().resid
        # 写入数据
        ex_df.loc[row[0]] = ex_val
    return ex_df


# 对行业以及传入的因子值做中性化
def neutralize_ind_factor(factor, ind, *args):
    ex_df = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for row in factor.iterrows():
        y = row[1]
        x = pd.get_dummies(ind.loc[row[0]].dropna())
        for f_ in args:
            if type(f_) == list:
                for f in f_:
                    x = pd.concat([x, f.loc[row[0]]], axis=1, join='inner')
            else:
                x = pd.concat([x, f_.loc[row[0]]], axis=1, join='inner')
        x = x.reindex(y.index)
        x = x.fillna(x.mean())
        ex_val = OLS(y, x, missing='drop').fit().resid
        ex_df.loc[row[0]] = ex_val
    return ex_df


# 对行业以及市值做中性化
def neutralize_ind_mv(factor, ind, mv):
    print("对因子进行市值，行业中性化处理")
    ex_df = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    lnmv = np.log(mv).replace([np.inf, -np.inf], np.nan)
    ex_df = neutralize_ind_factor(factor, ind, lnmv)
    return ex_df


def winsorize(df):
    md = df.median(axis=1)
    mad = (1.483 * (df.sub(md, axis=0)).abs().median(axis=1)).replace(0, np.nan)
    up = df.apply(lambda k: k > md + mad * 3)
    down = df.apply(lambda k: k < md - mad * 3)
    df[up] = df[up].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md + mad * 3, axis=0)
    df[down] = df[down].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md - mad * (0.5 + 3), axis=0)
    return df
