import pandas as pd
import numpy as np
from hqth_alpha.config import *
from datetime import datetime, timedelta, date
from datetime import datetime as dt
import matplotlib
import os
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib.backends.backend_pdf import PdfPages  # 输出图像的库
from collections import defaultdict
import seaborn as sns

sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('xtick', labelsize=50)
matplotlib.rc('ytick', labelsize=50)


# 加载环境数据
def load_env():
    print('加载环境数据')
    os.chdir(support_path)
    ex_ret_dict = {}
    with open('env_dict.pkl', 'rb') as f:
        env_dict = pkl.load(f)
    with open(support_path + r'\index_d_close_ts.pkl', 'rb') as f:
        indexes = pkl.load(f)
        # 计算相对不同基准的收益
        for index_name in ["上证50", "沪深300", "中证500"]:
            index_ret = indexes[index_name].pct_change(1, fill_method=None).shift(-2).dropna(how='all')
            index_ret.index = [dt.strptime(x, "%Y-%m-%d").date() for x in index_ret.index]
            ret = env_dict["future_retd"][1]
            ret = ret.reindex(index=index_ret.index)
            ex_ret = ret.sub(index_ret, axis=0).dropna(how='all')
            ex_ret_dict["相对" + index_name] = ex_ret
    ex_ret_dict["相对市场均值"] = env_dict["future_retd"][1].sub(env_dict["future_retd"][1].mean(axis=1), axis=0)
    return ex_ret_dict


# 重塑列名
def format_columns(event):
    columns_namesA = ["stock_code", "event_date"]
    columns_namesB = ["event_date", "stock_code"]
    if len(event.columns) != 2:
        print("不符合格式要求，请转换为两列！一列为股票代号，一列为事件发生时间。")
    elif len(event) == 0:
        print("事件为空！")
    sample_00 = event.iloc[0, 0]
    # 如果第一列第一行的元素是字符串类型
    if type(sample_00) == str:
        if len(event.iloc[0, 0]) <= 6:
            event.columns = columns_namesA
        else:
            event.columns = columns_namesB
            event = event.loc[:, ["stock_code", "event_date"]]
    # 如果是datetime类型
    elif type(sample_00) == datetime:
        event.columns = columns_namesB
        event = event.loc[:, ["stock_code", "event_date"]]
    # 如果是date类型
    elif type(sample_00) == date:
        event.columns = columns_namesB
        event = event.loc[:, ["stock_code", "event_date"]]
    # 如果是timestamp类型
    elif type(sample_00) == pd.Timestamp:
        event.columns = columns_namesB
        event = event.loc[:, ["stock_code", "event_date"]]
    else:
        try:
            # 如果是整形
            event.iloc[:, 0] = event.iloc[:, 0].apply(lambda x: int(str(x)))
            if sample_00 < 20000000:
                event.columns = columns_namesA
            else:
                event.columns = columns_namesB
                event = event.loc[:, ["stock_code", "event_date"]]
        except:
            print("数据格式有误！")
    return event


# 前部填补0至6位
def patch(x):
    while len(x) < 6:
        x = '0' + x
    return x


# 对事件数据进行清洗
def format_values(event):
    # 处理股票代码序列
    try:
        event.iloc[:, 0] = event.iloc[:, 0].apply(lambda x: str(x))
    except:
        print("股票代码数据格式有误！")
    sample_01 = event.iloc[0, 1]
    # 如果事件时间是str型
    if type(sample_01) == str:  # 可能是 '20200101' 或者 '2020-01-01' 或者 '2020 01 01'
        # 转换成datetime类型
        try:
            event.iloc[:, 1] = [datetime.strptime(str(int(x[:8])), '%Y%m%d').date() for x in event.iloc[:, 1]]
        except:
            try:
                event.iloc[:, 1] = [datetime.strptime(x[:10].strip(), '%Y-%m-%d').date() for x in event.iloc[:, 1]]
            except:
                try:
                    event.iloc[:, 1] = [datetime.strptime(x[:10].strip(), '%Y %m %d').date() for x in event.iloc[:, 1]]
                except:
                    print('时间格式有误！')
    # 如果本身就是datetime类型
    elif type(sample_01) == datetime:
        event.iloc[:, 1] = [x.date() for x in event.iloc[:, 1]]
    # 如果是timestamp类型
    elif type(sample_01) == pd.Timestamp:
        event.iloc[:, 1] = [x.to_pydatetime().date() for x in event.iloc[:, 1]]
    # 如果是date类型
    elif type(sample_01) == date:
        pass
    else:
        try:
            event.iloc[:, 1] = [datetime.strptime(str(int(x)), '%Y%m%d').date() for x in event.iloc[:, 1]]
        except:
            print('时间格式有误！')
    event = event[event.iloc[:, 0].apply(lambda x: len(x) <= 6)]
    # 填补小于6位的股票代码
    event.iloc[:, 0] = event.iloc[:, 0].apply(patch)
    event = event.sort_values(by="event_date").reset_index(drop=True)
    event = event.drop_duplicates()
    return event


# 清洗事件总函数
def format_event(event):
    if type(event) == pd.DataFrame:
        event = format_columns(event)
        event = format_values(event)
        return event
    elif type(event) == dict:
        event = pd.DataFrame(event)
        event = format_columns(event)
        event = format_values(event)
        return event
    else:
        print("请把数据调整成dataframe或者dict的形式！")


# 得到事件超额收益 index是股票代号，列是事件发生天数，value是每天的超额收益值
def get_ret_df(event, ex_ret_dict, periods):
    event_ret_df_dict = {}
    for index, ex_ret in ex_ret_dict.items():
        event_ret_list = []
        trade_dates = ex_ret.index
        for date in event['event_date'].unique():
            try:
                afterperiods = list(filter(lambda x: x >= date, trade_dates))[periods]
            except:
                continue
            code_list = list(set(event.loc[event['event_date'] == date, 'stock_code'].tolist()))
            sub_ret = ex_ret.loc[date:afterperiods].T.reindex(code_list)
            sub_ret.columns = list(range(0, periods + 1))
            event_ret_list.append(sub_ret)
        event_ret_df = pd.concat(event_ret_list).dropna()
        event_ret_df_dict[index] = event_ret_df
    return event_ret_df_dict


# 得到累计超额收益 是个series,index是时间，value是累计超额收益
def event_cum_ret(event_ret_df_dict):
    event_cum_ret_dict = {}
    for index, event_ret_df in event_ret_df_dict.items():
        mean_ret = event_ret_df.mean()
        event_cum_ret = (mean_ret + 1).cumprod()
        event_cum_ret = event_cum_ret / event_cum_ret[0]
        event_cum_ret_dict[index] = event_cum_ret
    return event_cum_ret_dict


# 计算超额收益总函数
def cal_return(event, ex_ret, periods):
    event_ret_df_dict = get_ret_df(event, ex_ret, periods)
    event_cum_ret_dict = event_cum_ret(event_ret_df_dict)
    return event_ret_df_dict, event_cum_ret_dict


# 事件描述性统计
def display_frequency(event, ex_ret_dict, pdf):
    valid_event = pd.merge(event, ex_ret_dict["相对市场均值"].T, left_on="stock_code",
                           right_on=ex_ret_dict["相对市场均值"].T.index).iloc[:, [0, 1]]

    event_freqdict_detail = defaultdict(dict)
    event_freqdict_year = defaultdict(dict)
    event_freqdict_month = defaultdict(dict)

    for i in valid_event["event_date"]:
        event_freqdict_detail[i.year][i.month] = event_freqdict_detail[i.year].get(i.month, 0) + 1
    for i in valid_event["event_date"]:
        event_freqdict_year[i.year] = event_freqdict_year.get(i.year, 0) + 1
    for i in valid_event["event_date"]:
        event_freqdict_month[i.month] = event_freqdict_month.get(i.month, 0) + 1

    event_freq_byyear = pd.DataFrame(event_freqdict_year, index=range(1)).T
    event_freq_byyear.columns = ["freq"]

    # 画按年统计的直方图
    plt.figure(figsize=(40, 30))
    plt.bar(event_freq_byyear.index, event_freq_byyear["freq"], width=0.8, edgecolor='red', color="yellow", lw=3)
    plt.xticks(np.arange(event_freq_byyear.index.min(), event_freq_byyear.index.max() + 1), rotation=30, fontsize=50)
    plt.yticks(fontsize=50)
    for x, y in zip(event_freq_byyear.index, event_freq_byyear["freq"]):
        plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
    plt.title("事件数量(逐年统计)", fontsize=70)
    plt.ylabel("频率", fontsize=60)
    plt.xlabel("年份", fontsize=60)
    pdf.savefig()

    event_freq_month = pd.DataFrame(event_freqdict_month, index=range(1)).T
    event_freq_month.columns = ["freq"]

    # 画按月统计的直方图
    plt.figure(figsize=(40, 30))
    plt.bar(event_freq_month.index, event_freq_month["freq"], width=0.8, edgecolor='red', color="yellow", lw=3)
    plt.xticks(np.arange(event_freq_month.index.min(), event_freq_month.index.max() + 1), rotation=30, fontsize=50)
    plt.yticks(fontsize=50)
    for x, y in zip(event_freq_month.index, event_freq_month["freq"]):
        plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
    plt.title("事件数量(逐月统计)", fontsize=70)
    plt.ylabel("频率", fontsize=60)
    plt.xlabel("月份", fontsize=60)
    pdf.savefig()

    # 画按年按月统计的直方图
    event_freq_detail = pd.DataFrame(event_freqdict_detail)
    event_freq_detail_fill = event_freq_detail.fillna(0).sort_index()
    year_ranges = len(event_freq_detail_fill.columns)
    event_freq_detail_fill = event_freq_detail_fill[np.sort(event_freq_detail_fill.columns)]
    cols = 4
    rows = year_ranges // cols
    if rows > 0:
        if rows > 1:
            fig, axes = plt.subplots(rows, cols, figsize=(70, 50))
            plt.suptitle("事件数量(逐年逐月统计)", fontsize=80)
            if (rows * cols) == year_ranges:
                for i in range(rows):
                    for j in range(cols):
                        axes[i, j].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i * cols + j])
                        axes[i, j].set_xticks(np.arange(1, 13))
                        axes[i, j].set_xlabel("月份", fontsize=50)
                        axes[i, j].set_ylabel("频率", fontsize=50)
                        axes[i, j].set_title(event_freq_detail_fill.iloc[:, i * cols + j].name, fontsize=60)
                        for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i * cols + j]):
                            axes[i, j].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                pdf.savefig()
            else:
                for i in range(rows):
                    for j in range(cols):
                        axes[i, j].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i * cols + j])
                        axes[i, j].set_xticks(np.arange(1, 13))
                        axes[i, j].set_xlabel("月份", fontsize=50)
                        axes[i, j].set_ylabel("频率", fontsize=50)
                        axes[i, j].set_title(event_freq_detail_fill.iloc[:, i * cols + j].name, fontsize=60)
                        for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i * cols + j]):
                            axes[i, j].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                pdf.savefig()
                fig, axes = plt.subplots(1, cols, figsize=(70, 50 / rows))
                for i in range((rows * cols), year_ranges):
                    axes[i - (rows * cols)].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i])
                    axes[i - (rows * cols)].set_xticks(np.arange(1, 13))
                    axes[i - (rows * cols)].set_xlabel("月份", fontsize=50)
                    axes[i - (rows * cols)].set_ylabel("频率", fontsize=50)
                    axes[i - (rows * cols)].set_title(event_freq_detail_fill.iloc[:, i].name, fontsize=60)
                    for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i]):
                        axes[i - (rows * cols)].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                for i in range(year_ranges - (rows * cols), cols):
                    plt.delaxes(axes[i])
                pdf.savefig()
        else:
            fig, axes = plt.subplots(1, cols, figsize=(70, 50 / cols + 5))
            plt.suptitle("事件数量(逐年逐月统计)", fontsize=80)
            if (cols) == year_ranges:
                for j in range(cols):
                    axes[j].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, j])
                    axes[j].set_xticks(np.arange(1, 13))
                    axes[j].set_xlabel("月份", fontsize=50)
                    axes[j].set_ylabel("频率", fontsize=50)
                    axes[j].set_title(event_freq_detail_fill.iloc[:, j].name, fontsize=60)
                    for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, j]):
                        axes[j].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                pdf.savefig()
            else:
                for j in range(cols):
                    axes[j].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, j])
                    axes[j].set_xticks(np.arange(1, 13))
                    axes[j].set_xlabel("月份", fontsize=50)
                    axes[j].set_ylabel("频率", fontsize=50)
                    axes[j].set_title(event_freq_detail_fill.iloc[:, j].name, fontsize=60)
                    for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, j]):
                        axes[j].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                pdf.savefig()
                fig, axes = plt.subplots(1, cols, figsize=(70, 50 / cols + 5))
                for i in range((cols), year_ranges):
                    axes[i - cols].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i])
                    axes[i - cols].set_xticks(np.arange(1, 13))
                    axes[i - cols].set_xlabel("月份", fontsize=50)
                    axes[i - cols].set_ylabel("频率", fontsize=50)
                    axes[i - cols].set_title(event_freq_detail_fill.iloc[:, i].name, fontsize=60)
                    for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i]):
                        axes[i - cols].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
                for i in range(year_ranges - (cols), cols):
                    plt.delaxes(axes[i])
                pdf.savefig()
    else:
        fig, axes = plt.subplots(1, cols, figsize=(70, 50 / cols + 5))
        plt.suptitle("事件数量(逐年逐月统计)", fontsize=80)
        for i in range(year_ranges):
            axes[i].bar(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i])
            axes[i].set_xticks(np.arange(1, 13))
            axes[i].set_xlabel("月份", fontsize=50)
            axes[i].set_ylabel("频率", fontsize=50)
            axes[i].set_title(event_freq_detail_fill.iloc[:, i].name, fontsize=60)
            for x, y in zip(event_freq_detail_fill.index, event_freq_detail_fill.iloc[:, i]):
                axes[i].text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=45)
        for i in range(year_ranges, cols):
            plt.delaxes(axes[i])
        pdf.savefig()

    # 画各年事件占比的直方图
    plt.figure(figsize=(70, 50))
    labels = range(1, 13)
    data = list()
    event_freq_detail_fill = event_freq_detail_fill.reindex(index=range(1, 13))
    for i in event_freq_detail_fill.columns:
        data.append(event_freq_detail_fill.loc[:, i].values)
    x = range(len(labels))
    width = 0.7
    bottom_y = np.zeros(len(labels))
    data = np.array(data)
    sums = np.sum(data, axis=0)
    years = event_freq_detail_fill.columns
    for i in range(len(data)):
        y = data[i] / sums
        plt.bar(x, y, width, bottom=bottom_y, label=years[i])
        for m, p, b in zip(x, y, bottom_y):
            if p > 0:
                plt.text(m, b + p / 4, '%.2f' % p, ha='center', va="bottom", fontsize=60)
        bottom_y = y + bottom_y
        bottom_y[np.isnan(bottom_y)] = 0
    plt.legend(loc='best', fontsize=50)
    plt.xticks(x, labels, fontsize=90)
    plt.yticks(np.linspace(0, 1, 6), fontsize=90)
    plt.title('事件各年占比', fontsize=110)
    plt.xlabel("月份", fontsize=100)
    pdf.savefig()
    plt.show()


# 画事件超额收益折线图
def display_return(event_cum_ret_dict, periods, pdf):
    plt.figure(figsize=(50, 40))
    plt.xticks(range(0, periods + 1, 5), fontsize=60)
    plt.yticks(np.linspace(0.9, 1.1, 41), fontsize=60)
    plt.title("事件累计超额收益", fontsize=90)
    plt.xlabel("时间", fontsize=70)
    plt.ylabel("超额收益率", fontsize=70)
    for index, event_cum_ret in event_cum_ret_dict.items():
        plt.plot(event_cum_ret, label=index, linewidth=6)
    plt.legend(loc='best', fontsize=50)
    pdf.savefig()
    plt.show()


# 画事件超额收益T值图
def cal_tvalue(event_ret_df_dict, periods, pdf):
    for index, event_ret_df in event_ret_df_dict.items():
        means = (event_ret_df + 1).cumprod(axis=1).mean()
        stds = (event_ret_df + 1).cumprod(axis=1).std()
        t_values = np.sqrt(len(event_ret_df)) * ((means - 1) / stds)
        plt.figure(figsize=(50, 40))
        plt.bar(range(0, periods + 1), t_values)
        plt.yticks(fontsize=50)
        plt.xticks(np.linspace(0, periods, periods // 5 + 1), fontsize=50, rotation=30)
        plt.xlabel("时间", fontsize=60)
        plt.ylabel("T值", fontsize=60)
        plt.title(f"事件T检验({index})", fontsize=80)
        pdf.savefig()
        plt.show()


# 画按月胜率分布热力图
def cal_win_ratio_bymonth_heatmap(event, ex_ret_dict, periods, pdf):
    print("正在计算每个月的胜率！")
    for index, ex_ret in ex_ret_dict.items():
        date_bymonth = defaultdict(dict)
        for date in event['event_date'].unique():
            date_bymonth[date.year][date.month] = date_bymonth[date.year].get(date.month, [])
        for date in event["event_date"].unique():
            date_bymonth[date.year][date.month].append(date)
        win_bymonth = defaultdict(dict)
        for year in date_bymonth.keys():
            for month in date_bymonth[year].keys():
                event_ret_list = []
                trade_dates = ex_ret.index
                for date in date_bymonth[year][month]:
                    try:
                        afterperiods = list(filter(lambda x: x >= date, trade_dates))[periods]
                    except:
                        continue
                    code_list = list(set(event.loc[event['event_date'] == date, 'stock_code'].tolist()))
                    sub_ret = ex_ret.loc[date:afterperiods].T.reindex(code_list)
                    sub_ret.columns = list(range(0, periods + 1))
                    event_ret_list.append(sub_ret)
                if len(event_ret_list) > 0:
                    event_ret_df = pd.concat(event_ret_list).dropna()
                    win_ratio = np.sum(
                        (event_ret_df + 1).cumprod(axis=1).iloc[:, periods] > ((event_ret_df + 1).iloc[:, 0])) / len(
                        event_ret_df)
                    win_bymonth[year][month] = win_ratio
        win_bymonth_frame = pd.DataFrame(win_bymonth).sort_index().reindex(index=range(1, 13))
        fig = plt.figure(figsize=(50, 50))
        sns_plot = sns.heatmap(win_bymonth_frame, cmap='YlGnBu', annot=True, annot_kws={'size': 50})
        sns_plot.tick_params(labelsize=60, direction='in')
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=60, direction='in', top='off', bottom='off', left='off', right='off')
        plt.title(f"胜率({index})", fontsize=80)
        pdf.savefig()
        plt.show()


# 画按年胜率分布图
def cal_win_ratio_byyear(event, ex_ret_dict, periods, pdf):
    print("正在计算每年的胜率！")
    fig, axes = plt.subplots(4, 1, figsize=(50, 32))
    cur_row = 0
    for index, ex_ret in ex_ret_dict.items():
        date_byyear = {i.year: [] for i in event["event_date"].unique()}
        for date in event['event_date'].unique():
            date_byyear[date.year].append(date)
        win_byyear = {}
        for year in date_byyear.keys():
            event_ret_list = []
            trade_dates = ex_ret.index
            for date in date_byyear[year]:
                try:
                    afterperiods = list(filter(lambda x: x >= date, trade_dates))[periods]
                except:
                    continue
                code_list = list(set(event.loc[event['event_date'] == date, 'stock_code'].tolist()))
                sub_ret = ex_ret.loc[date:afterperiods].T.reindex(code_list)
                sub_ret.columns = list(range(0, periods + 1))
                event_ret_list.append(sub_ret)
            if len(event_ret_list) > 0:
                event_ret_df = pd.concat(event_ret_list).dropna()
                win_ratio = np.sum(
                    (event_ret_df + 1).cumprod(axis=1).iloc[:, periods] > ((event_ret_df + 1).iloc[:, 0])) / len(
                    event_ret_df)
                win_byyear[year] = win_ratio
        win_byyear_frame = pd.Series(win_byyear).to_frame()
        win_byyear_frame = win_byyear_frame.rename(columns={0: "胜率"}).T
        table_width = len(win_byyear_frame.columns)
        tb = axes[cur_row].table(win_byyear_frame.round(4).values, cellLoc='center',
                                 colLabels=win_byyear_frame.columns,
                                 rowLabels=win_byyear_frame.index,
                                 rowColours=plt.cm.BuPu(np.linspace(0, 0.5, len(win_byyear_frame.index)))[::-1],
                                 colColours=plt.cm.plasma(np.linspace(0.1, 1.5, table_width)),
                                 loc='center right',
                                 colWidths=[1 / (table_width + 1)] * table_width)
        tb.scale(1.2, 3.5 * table_width)
        tb.set_fontsize(70)
        axes[cur_row].set_title(f'每年的胜率({index})', fontsize=80)
        axes[cur_row].axis('off')
        cur_row += 1
    pdf.savefig()
    plt.show()


# 计算总胜率
def cal_win_ratio_total(event, ex_ret_dict, periods):
    print("正在计算总胜率！")
    for index, ex_ret in ex_ret_dict.items():
        event_ret_list = []
        trade_dates = ex_ret.index
        for date in event['event_date'].unique():
            try:
                afterperiods = list(filter(lambda x: x >= date, trade_dates))[periods]
            except:
                continue
            code_list = list(set(event.loc[event['event_date'] == date, 'stock_code'].tolist()))
            sub_ret = ex_ret.loc[date:afterperiods].T.reindex(code_list)
            sub_ret.columns = list(range(0, periods + 1))
            event_ret_list.append(sub_ret)
        event_ret_df = pd.concat(event_ret_list).dropna()
        win_ratio = np.sum((event_ret_df + 1).cumprod(axis=1).iloc[:, periods] > ((event_ret_df + 1).iloc[:, 0])) / len(
            event_ret_df)
        print(f"总胜率({index})为{win_ratio:.4f}!")


# 事件回测框架总函数
def event_analyse(event, event_name=None, periods=60):
    if event_name == None:
        event_name = input('请命名事件：')
    while (event_name == ''):
        print('未检测到命名，请重新输入：')
        event_name = input()
    file_name = event_name + '事件报告.pdf'
    print('当前因子报告默认输出路径：%s\\' % (event_output_path), file_name)
    # 将事件评价结果输出至pdf
    with PdfPages(os.path.join(event_output_path, file_name)) as pdf:
        ex_ret_dict = load_env()
        print('清洗事件数据!')
        event = format_event(event)
        event_ret_df_dict, event_cum_dict = cal_return(event, ex_ret_dict, periods)
        display_frequency(event, ex_ret_dict, pdf)
        display_return(event_cum_dict, periods, pdf)
        cal_tvalue(event_ret_df_dict, periods, pdf)
        cal_win_ratio_total(event, ex_ret_dict, periods)
        cal_win_ratio_byyear(event, ex_ret_dict, periods, pdf)
        cal_win_ratio_bymonth_heatmap(event, ex_ret_dict, periods, pdf)
