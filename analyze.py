import pandas as pd
import numpy as np
from hqth_alpha.config import *
import hqth_alpha.data as hd
import hqth_alpha.process as hp
from datatime import datetime, timedelta, date
import os
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 15


def create_env(index_name=None):
    p_close = hd.load_ohlc_data("CLOSE_PRICE_2")
    p_open0 = hd.load_ohlc_data("OPEN_PRICE").replace(0, np.nan)
    p_open = hd.load_ohlc_data("OPEN_PRICE_2").replace(0, np.nan)
    p_open[p_open.isnull()] = p_close.shift()[p_open.isnull()]

    up_limit = hd.load_ohlc_data('LIMIT_UP_PRICE').T.reindex(p_open.columns).T
    down_limit = hd.load_ohlc_data('LIMIT_DOWN_PRICE').T.reindex(p_open.columns).T

    trade_status = hd.load_support_data('trade_status')
    st_status = hd.load_support_data('st_status')
    newstock = trade_status.isnull().astype('int').rolling(60, min_period=1).max() == 1

    # 计算不同期限的未来收益率
    future_retd = {}
    for period in [1, 5, 10, 20, 40]:
        future_retd[period] = p_open.pct_change(period, fill_method=None).shift(-period - 1).dropna(how='all')

        if index_name == None:
            invalid_data = (trade_status != 'L') | (st_status == 1) | newstock | (p_open0 == up_limit) | (
                    p_open0 == down_limit)
        elif index_name in ["中证500", "上证50", "沪深300", "国证a指"]:
            index_member = hd.load_support_data(index_name + "成份_ts").drop_duplicates(subset=['date', 'code'])
            index_member_df = index_member.pivot(index='date', columns='code', values='rank')
            index_member_df.index = pd.to_datetime(index_member_df.index.to_series().astype(str)). \
                apply(lambda x: x.to_pydatetime().date())
            index_member_df.columns = [x[2:] for x in index_member_df.columns]
            index_member_df = index_member_df.reindex(p_open.index).T.reindex(p_open.columns).T.fillna(0)
            p_open[index_member_df == 0] = np.nan
            invalid_data = (trade_status != 'L') | (st_status == 1) | newstock | (p_open0 == up_limit) | (
                        p_open0 == down_limit) | (index_member_df == 0)
        env_dict = {'open': p_open, 'invalid': invalid_data, 'future_retd': future_retd}
        return env_dict


def load_env(index_name=None):
    print("加载环境数据")
    os.chdir(support_path)
    if index_name is None:
        with open('env_dict.pkl', 'rb') as f:
            env_dict = pkl.load(f)
    else:
        with open('env_dict' + index_name + 'pkl', 'rb') as f:
            env_dict = pkl.load(f)
    return env_dict


def save_env(env_dict, index_name=None):
    os.chdir(support_path)
    if index_name is None:
        with open('env_dict.pkl', 'wb') as f:
            pkl.dump(env_dict, f)
    else:
        with open('env_dict' + index_name + 'pkl', 'wb') as f:
            pkl.dump(env_dict, f)


def create_factor(factor, env_dict, holding_periods):
    "清洗因子数据并匹配收益率"
    print("清洗因子数据并匹配收益率")
    factor = hp.format_factor(factor)
    if 1 not in holding_periods: holding_periods = [1] + holding_periods
    holding_periods = sorted(holding_periods)

    factor = factor.replace([np.inf, -np.inf], np.nan)

    factor2 = factor.dropna(how='all').reindex(env_dict['open'].index).T.reindex(env_dict['open'].columns).T
    cover_series = factor2[env_dict['open'].notnull()].count(axis=1) / env_dict['open'].notnull().sum(axis=1)

    factor2 = hp.winsorize(factor2)

    out = env_dict['invalid'].shift(-1).fillna(True)
    factor[out] = np.nan

    # 计算不同期限的未来收益率
    future_retd_new = {}
    for period in holding_periods:
        if period in env_dict["future_retd"].keys():
            future_retd_new[period] = env_dict['future_retd'][period]
        else:
            future_retd_new[period] = env_dict['open'].pct_change(period, fill_method=None).shift(-period - 1).dropna(
                how='all')

        factor2 = factor2.reindex(future_retd_new[holding_periods][-1].index).dropna(how='all')
        tot_dict = {'factor': factor2, 'future_retd': future_retd_new, 'holding_periods': holding_periods,
                    'cover': cover_series}
        return tot_dict


# 因子描述统计
def display_factor(tot_dict, pdf):
    '''
    数据统计
    因子覆盖度
    因子分布直方图
    '''
    print("因子分布")
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    tot_dict['cover'].plot(title='cover', grid=True)
    plt.subplot(1, 2, 2)
    tot_dict['factor'].stack().dropna().hist(bins=100)
    plt.title('distribution')

    pdf.savefig()
    plt.show()


def cal_tot_ic(factor, f_ret):
    '''计算整体IC'''
    tmp_ret = f_ret.copy()
    tmp_factor = factor.copy().reindex(tmp_ret.index).reindex(tmp_ret.columns, axis='columns')

    tmp_factor[np.isnan(tmp_ret)] = np.nan
    tmp_ret[np.isnan(tmp_factor)] = np.nan

    tmp_factor -= tmp_factor.sum().sum() / tmp_factor.notnull().sum().sum()
    tmp_ret -= tmp_ret.sum().sum() / tmp_ret.notnull().sum().sum()

    tmp_ic = np.nansum((tmp_factor.values * tmp_ret.values)) / \
             np.sqrt((np.nansum((tmp_factor.values ** 2)) * np.nansum((tmp_ret.values ** 2))))
    return tmp_ic


def cal_ic(tot_dict, pdf, plotting=True):
    print("因子IC分析")
    ic_list = []
    ic_adjusted_list = []
    ic_tot_list = []
    for period in tot_dict["holding_periods"]:
        ic_list.append(tot_dict['factor'].apply(lambda x: x.corr(tot_dict['future_retd'][period].loc[x.name]), axis=1))
        ic_tot_list.append(cal_tot_ic(tot_dict['factor'], tot_dict['future_retd'][period]))
        ic_adjusted_list.append(tot_dict['factor'].apply(lambda x: x.corr(tot_dict['future_retd'][period].loc[x.name]) \
                                                                   * tot_dict['future_retd'][period].loc[
                                                                       x.name].std() * 2000 / period, axis=1))

    ic_df = pd.concat(ic_list, axis=1, keys=tot_dict['holding_periods'].dropna(how='all'))
    ic_adjusted_df = pd.concat(ic_adjusted_list, axis=1, keys=tot_dict['holding_periods'].dropna(how='all'))
    ic_tot = pd.Series(ic_tot_list, index=tot_dict['holding_periods'])

    ic_summary_table = pd.cancat(
        [ic_df.mean(), ic_df.std(), np.sqrt(len(ic_df)) * (ic_df.mean() / ic_df.std()), ic_adjusted_df.mean(), ic_tot], \
        axis=1, keys=['IC mean', 'IC std', 'IC T-Statistics', 'IC adj month', 'IC tot']).T

    print(ic_summary_table)
    if plotting:
        table_width = len(ic_summary_table.columns)
        plt.figure(figsize=(20, 8))
        tb = plt.table(ic_summary_table.round(4).values, cellLoc='center',
                       rowLabels=ic_summary_table.index.to_list(),
                       colLabels=ic_summary_table.columns,
                       rowColours=plt.cm.BuPu(np.linspace(0, 0.5, len(ic_summary_table.index)))[::-1],
                       colColours=plt.cm.plasma(np.linspace(0.1, 1.5, table_width)),
                       loc='center_right',
                       colWidths=[1 / (table_width + 1)] * table_width)
        tb.scale(1, 0.5 * table_width)
        tb.set_fontsize(15)
        plt.tight_layout()
        plt.axis('off')
        plt.title('不同时段因子值的IC表现')
        pdf.savefig()

        ic_df.rolling(21).mean().shift(-11).plot(title='monthly mean IC', kind='area', stacked=False, figsize=(20, 10),
                                                 colormap='plasma', grid=True)

        pdf.savefig()
        plt.show()

        return ic_df


def cal_group_ret(tot_dict, pdf, n=10):
    print("因子分组分析")
    group = round(tot_dict['factor'].rank(pct=True, axis=1) * 0.9999 * n + 0.5)
    mean_ret_dict = {}
    for period in tot_dict['holding_periods']:
        mean_ret_dict[period] = group.apply(
            lambda x: tot_dict['future_retd'][period].loc[x.name].groupby(x).mean() / period, axis=1)
        mean_ret_df = pd.concat([21 * (mean_ret_dict[p].mean()) for p in tot_dict['holding_periods']], axis=1,
                                keys=tot_dict['holding_periods'])
        mean_ret_df.plot(kind='bar', figsize=(20, 8), title='monthly mean return per group')
        pdf.savefig()
        plt.show()

    long_short_nvs = pd.concat(
        [(mean_ret_dict[p].iloc[:, -1] - mean_ret_dict[p].iloc[:, 0]).cumsum() for p in tot_dict['holding_periods']],
        axis=1, keys=tot_dict['holding_periods'])
    long_short_nvs.plot(title='long short net value', figsize=(20, 10), colormap='plasma', grid=True)
    pdf.savefig()
    plt.show()

    for period in tot_dict['holding_periods']:
        group_nvs = mean_ret_dict[period].fillna(0).cumsum()
        group_nvs.plot(title='group net value for period'+str(period),figsize=(20,10),colormap='plasma',grid=True)
        pdf.savefig()
        plt.show()

    return mean_ret_dict

def analyse(factor,factor_name=None,periods=[1,5,10,20],n=10,index_name=None):
    if factor_name == None:
        factor_name = input("请命名因子：")
    while (factor_name==''):
        print("未检测到命名，请重新输入：")
        factor_name = input()

    if index_name == None:
        file_name = factor_name + '因子报告.pdf'
    else:
        file_name = factor_name +'(' + index_name + ')因子报告.pdf'

    print('当前因子报告默认输出路径:%s\\'%(factor_output_path),file_name)

    with PdfPages(os.path.join(factor_output_path,file_name)) as pdf:
        env_dict = load_env(index_name)
        tot_dict = create_factor(factor,env_dict,periods)
        display_factor(tot_dict,pdf)
        ic_df = cal_ic(tot_dict,pdf)
        gr_dict = cal_group_ret(tot_dict,pdf,n)
        ans_dict = {'ic_df':ic_df,'gr_dict':gr_dict}
    return ans_dict

def analyse_ic(factor,periods=[1,5,10,20],index_name=None):
    env_dict = load_env(index_name)
    tot_dict = create_factor(factor,env_dict,periods)
    ic_df = cal_ic(tot_dict,plotting=False)
    return ic_df

