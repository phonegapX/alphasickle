# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from collections import defaultdict
from copy import deepcopy
from scipy.optimize import linprog
from factor_generate import FactorGenerater
from factor_preprocess import info_cols
from single_factor_test import *

__all__ = ['index_enhance_model', 'get_factor', 'get_stock_wt_in_index', 'factor_process']

sns.set(style="darkgrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    #正常显示负号
plt.rcParams['figure.figsize'] = (16.0, 9.0)  #图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 15                #字体大小

#工作目录，存放代码
work_dir = os.path.dirname(os.path.dirname(__file__))
#经过预处理后的因子截面数据存放目录
factor_panel_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
#原始经过预处理的因子的矩阵数据存放目录
factor_matrix_path = os.path.join(work_dir, '单因子检验', '因子矩阵')
#合成、正交因子存放目录（如无则自动生成）
rm_save_path = os.path.join(work_dir, '收益模型')
#测试结果图表存放目录（如无则自动生成）
index_enhance_dir = os.path.join(work_dir, '指数增强模型')

industry_benchmark = 'zx'      #中信一级行业

#自动生成合成、正交因子存放目录
if not os.path.exists(rm_save_path):
    os.mkdir(rm_save_path)
#自动生成指数增强模型结果存放目录
if not os.path.exists(index_enhance_dir):
    os.mkdir(index_enhance_dir)

factor_generater = FactorGenerater()

def get_stock_wt_in_index(index):
    """
    获取指数（000300.SH或000905.SH）中各截面期成分股所占权重
    """
    global factor_generater
    if index.startswith('000300') or index.startswith('399300'):
        index_wt = factor_generater.hs300_wt
    elif index.startswith('000905') or index.startswith('399905'):
        index_wt = factor_generater.zz500_wt
    else:
        msg = f'暂不支持当前指数：{index}'
        raise Exception(msg)
    return index_wt

def get_factor_corr(factors=None, codes=None):
    """
    计算因子相关系数
    """
    if factors is None:
        factors = get_factor_names()
    factors_matrix_dat = get_factor(factors)
    factors_panel_dat = concat_factors_panel(factors, factors_matrix_dat, codes, False, False)
    corrs = []
    for date in sorted(factors_panel_dat.keys()):
        factor_panel = factors_panel_dat[date]
        corrs.append(factor_panel.corr())

    avg_corr = reduce(lambda df1, df2: df1 + df2, corrs) / len(corrs)
    return avg_corr

def plot_corr_heatmap(corr, method, preprocessed=True):
    """
    绘制相关系数热力图
    """
    global rm_save_path
    corrfig_path = os.path.join(rm_save_path, '相关系数图')
    if method == 's':
        mid_path = os.path.join(corrfig_path, '分层抽样')
    elif method == 'l':
        mid_path = os.path.join(corrfig_path, '线性规划')
    else:
        mid_path = rm_save_path

    if preprocessed:
        save_path = os.path.join(mid_path, '处理后')
    else:
        save_path = os.path.join(mid_path, '处理前')

    if not os.path.exists(corrfig_path):
        os.mkdir(corrfig_path)
        os.mkdir(mid_path)
        os.mkdir(save_path)
        save_num = 1
    else:
        if not os.path.exists(mid_path):
            os.mkdir(mid_path)
        if os.path.exists(save_path):
            try:
                save_num = sorted(int(f for f in os.listdir(save_path)))[-1] + 1
            except:
                save_num = 1
        else:
            os.mkdir(save_path)
            save_num = 1

    final_path = os.path.join(save_path, f'{save_num}.jpg')

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(corr, 
            linewidths=0.05, 
            vmin=-1, 
            vmax=1, 
            annot=True, 
            cmap='rainbow')
    plt.xticks(rotation=30) #rotate to avoid overlap text.
    fig.savefig(final_path)
    plt.close()

def factor_concat(factors_to_concat, new_factor_name, weight=None):
    """
    因子合成：
    输入：待合并因子的名称(,分隔); 合成后的因子存储名称(自动添加_con后缀); 合成权重(默认等权)
    输出：合成后因子的因子截面数据和矩阵数据
    """
    global factor_panel_path, rm_save_path, info_cols
    if not new_factor_name.endswith('con'):
        new_factor_name += '_con'
    cfactor_spath = os.path.join(rm_save_path, '新合成因子')
    cpanel_spath = os.path.join(cfactor_spath, '因子截面')
    cmatrix_spath = os.path.join(cfactor_spath, '因子矩阵')
    if not os.path.exists(cfactor_spath):
        os.mkdir(cfactor_spath)
        os.mkdir(cpanel_spath)
        os.mkdir(cmatrix_spath)

    if ',' in factors_to_concat:
        factors_to_concat = factors_to_concat.split(',')

    if weight is None:
        apply_func = np.mean
        col_name = new_factor_name+'_equal'
    else:
        apply_func = lambda df: np.sum(weight*df)
        col_name = new_factor_name

    if os.path.exists(os.path.join(cmatrix_spath, col_name+'.csv')):
        print(f'{col_name}因子数据已存在')
        return

    panelfactors = os.listdir(cpanel_spath)

    for f in os.listdir(factor_panel_path):
        dat = pd.read_csv(os.path.join(factor_panel_path, f), encoding='gbk', engine='python', index_col=[0])
        factor_dat = dat[factors_to_concat]
        factor_concated = factor_dat.apply(apply_func, axis=1)
        factor_concated.name = col_name
        if panelfactors: #判断目标文件是否存在,存在就只需更新内容,故意不写成 if f in panelfactors,为了就是能提早发现错误
            panel_dat = pd.read_csv(os.path.join(cpanel_spath, f), encoding='gbk', engine='python', index_col=[0])
            if col_name in panel_dat.columns:
                del panel_dat[col_name]
            panel_dat = pd.concat([panel_dat, factor_concated], axis=1)
        else:
            panel_dat = pd.concat([dat[info_cols], factor_concated], axis=1)

        panel_dat.to_csv(os.path.join(cpanel_spath, f), encoding='gbk')

    panel_to_matrix([col_name], factor_path=cpanel_spath, save_path=cmatrix_spath) #将刚生成的截面数据转换成矩阵数据存放到cmatrix_spath目录当中
    print(f"创建{col_name}因子数据成功.")

def orthogonalize(factors_y, factors_x, codes=None, index_wt=None):
    """
    因子正交：
    输入：因变量(y)、自变量(x)因子名称（,分隔），类型：字符串
    输出：经过正交的因子截面数据和因子矩阵数据
    """
    global rm_save_path, factor_panel_path, info_cols
    ofactor_spath = os.path.join(rm_save_path, '正交后因子')
    opanel_spath = os.path.join(ofactor_spath, '因子截面')
    omatrix_spath = os.path.join(ofactor_spath, '因子矩阵')
    if not os.path.exists(ofactor_spath):
        os.mkdir(ofactor_spath)
        os.mkdir(opanel_spath)
        os.mkdir(omatrix_spath)

    for fac in factors_y.copy():
        if os.path.exists(os.path.join(omatrix_spath, fac+'_ortho.csv')):
            print(f'{fac}_ortho因子数据已存在')
            factors_y.remove(fac)

    if len(factors_y) == 0:
        return

    panel_y = concat_factors_panel(factors_y, codes=codes, ind=False, mktcap=False)
    panel_x = concat_factors_panel(factors_x, codes=codes, ind=False, mktcap=False)

    ortho_y = {}
    for date in sorted(panel_x.keys()):
        y = panel_y[date]
        X = panel_x[date]
        cur_index_wt = index_wt[date].dropna()

        data_to_regress = pd.concat([X, y], axis=1)
        mut_index = data_to_regress.index.intersection(cur_index_wt.index)
        data_to_regress = data_to_regress.loc[mut_index, :]
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        cut_loc = len(y.columns)
        X, ys = data_to_regress.iloc[:, :-cut_loc], data_to_regress.iloc[:, -cut_loc:]

        resids = pd.DataFrame()
        #params_a = pd.DataFrame()
        for fac in ys.columns:
            y = ys[fac]
            _, params, resid_y = regress(y, X, intercept=True)
            #params_a = pd.concat([params_a, params], axis=1)
            resid_y.name = fac + '_ortho'
            resids = pd.concat([resids, resid_y], axis=1)
        ortho_y[date] = resids

    for date in ortho_y.keys():
        date_str = str(date)[:10]
        cur_panel_ortho = ortho_y[date]
        basic_info = pd.read_csv(os.path.join(factor_panel_path, date_str+'.csv'), encoding='gbk', engine='python', index_col=[0])[info_cols]
        new_panel = pd.merge(basic_info, cur_panel_ortho, left_on='code', right_index=True)
        new_panel.to_csv(os.path.join(opanel_spath, date_str+'.csv'), encoding='gbk')

    factors_ortho = [fac+'_ortho' for fac in factors_y]
    panel_to_matrix(factors_ortho, factor_path=opanel_spath, save_path=omatrix_spath) #将刚生成的截面数据转换成矩阵数据存放到omatrix_spath目录当中
    print(f"创建{','.join(factors_ortho)}因子数据成功.")

def get_panel_data(names, fpath, codes=None):
    res = defaultdict(pd.DataFrame)
    if not isinstance(names, list):
        names = [names]
    for file in os.listdir(fpath):
        date = pd.to_datetime(file.split('.')[0])
        datdf = pd.read_csv(os.path.join(fpath, file), encoding='gbk', engine='python', index_col=['code'])
        for name in names:
            dat = datdf.loc[:, name]
            dat.name = date
            if codes is not None:
                dat = dat.loc[codes]
            res[name] = pd.concat([res[name], dat], axis=1)
    return res

def get_matrix_data(name, fpath, codes=None):
    data = pd.read_csv(os.path.join(fpath, name+'.csv'), encoding='gbk', engine='python', index_col=[0])
    data.columns = pd.to_datetime(data.columns)
    if codes is not None:
        data = data.loc[codes, :]
    return {name: data}

def get_factor(factor_names, codes=None):
    """
    获取指定因子全部数据（仅预处理、合成、正交）
    """
    #指定因子所在因子路径
    factor_paths = [(f, get_factor_path(f)) for f in factor_names]
    #矩阵形式保存因子的路径
    factors_matrix = {fname: path for fname, path in factor_paths if path.endswith('因子矩阵')}
    #截面形式保存因子的路径
    factors_panel = defaultdict(list)
    for fname, path in factor_paths:
        if path.endswith('截面') or '预处理' in path:
            factors_panel[path].append(fname)
    #读取矩阵形式保存的因子
    res = {}
    for fname, fpath in factors_matrix.items():
        res.update(get_matrix_data(fname, fpath, codes))
    #读取截面形式保存的因子
    for fpath, fnames in factors_panel.items():
        res.update(get_panel_data(fnames, fpath, codes))
    return res #{因子名: dataframe(因子矩阵数据)}

def get_factor_path(factor_name, frame='matrix'):
    """
    根据因子名称后缀，识别因子路径（仅预处理、合成、正交）
    """
    global factor_panel_path, rm_save_path, factor_matrix_path, info_cols, industry_benchmark
    new_concated_spath = os.path.join(rm_save_path, '新合成因子')
    orthoed_spath = os.path.join(rm_save_path, '正交后因子')

    basic_infos = [name for name in info_cols if name not in ('MKT_CAP_FLOAT', f'industry_{industry_benchmark}', 'PCT_CHG_NM')]
    if factor_name in basic_infos:
        return factor_panel_path

    if factor_name.endswith('_con') or factor_name.endswith('_con_equal'):
        new_concated = True
    else:
        new_concated = False
        if factor_name.endswith('_ortho'):
            orthoed = True
        else:
            orthoed = False

    if frame == 'panel':
        if new_concated:
            open_path = os.path.join(new_concated_spath, '因子截面')
        elif orthoed:
            open_path = os.path.join(orthoed_spath, '因子截面')
        else:
            open_path = factor_panel_path

    elif frame == 'matrix':
        if new_concated:
            open_path = os.path.join(new_concated_spath, '因子矩阵')
        elif orthoed:
            open_path = os.path.join(orthoed_spath, '因子矩阵')
        else:
            open_path = factor_matrix_path
    else:
        raise TypeError(f"不支持的因子数据格式：{frame}")
    return open_path

def concat_factors_panel(factors=None, factors_dict=None, codes=None, ind=True, mktcap=True):
    """
    将一个或者多个因子矩阵数据转换为因子截面数据,按需可以加入行业伪变量和市值对数
    """
    global industry_benchmark
    factors = deepcopy(factors)
    if factors:
        if isinstance(factors, str):
            factors = factors.split(',')
    else:
        factors = []

    if ind:
        factors.append(f'industry_{industry_benchmark}')
    if mktcap:
        factors.append('MKT_CAP_FLOAT')

    if codes is not None and factors_dict is not None:
        factors_dict = {fac: datdf.loc[codes,:] for fac, datdf in factors_dict.items()}

    if (factors_dict is None) or ('MKT_CAP_FLOAT' in factors) or (f'industry_{industry_benchmark}' in factors):
        matrix = {}
        for fac in factors:
            fpath = get_factor_path(fac)
            matrix.update(get_matrix_data(fac, fpath, codes))
        if factors_dict:
            matrix.update(factors_dict)
    else:
        matrix = factors_dict

    panel = defaultdict(pd.DataFrame)

    #对每个时间截面，合并因子数据
    facs = sorted(matrix.keys())
    for fac in facs:
        for date in matrix[fac]:
            cur_fac_panel_data = matrix[fac][date]
            cur_fac_panel_data.name = fac
            if 'industry' in fac and (ind == True):
                cur_fac_panel_data = pd.get_dummies(cur_fac_panel_data)
            elif fac == 'MKT_CAP_FLOAT' and (mktcap == True):
                cur_fac_panel_data = np.log(cur_fac_panel_data)
                cur_fac_panel_data.name = 'ln_mkt_cap'

            panel[date] = pd.concat([panel[date], cur_fac_panel_data], axis=1)

    return panel

def get_exponential_weights(window=12, half_life=6):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] 

def wt_sum(series, wt):
    if len(series) < len(wt):
        return np.sum(series * wt[:len(series)] / np.sum(wt[:len(series)]))
    else:
        return np.sum(series * wt / np.sum(wt))

def factor_return_forecast(factors_x, factor_data=None, window=12, half_life=6):
    """
    因子收益预测：
    输入：自变量(x)因子名称（,分隔），类型：字符串
    输出：截面回归得到的因子收益率预测值，行：因子名称，列：截面回归当期日期
    """
    index_wt = get_stock_wt_in_index('000300.SH')
    ret_matrix = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']

    if factor_data is None:
        panel_x = concat_factors_panel(factors_x)
    else:
        panel_x = factor_data

    #逐期进行截面回归，获取回归系数，作为因子收益
    factor_rets = pd.DataFrame()
    for date in sorted(panel_x.keys()):
        y = ret_matrix[date] #下期(期末)股票收益率
        X = panel_x[date]    #当期(期末)因子值(因子暴露,因子载荷),也就是下期期初的值
        cur_index_wt = index_wt[date].dropna()

        data_to_regress = pd.concat([X, y], axis=1)
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        mut_index = data_to_regress.index.intersection(cur_index_wt.index)
        data_to_regress = data_to_regress.loc[mut_index, :]
        X, y = data_to_regress.iloc[:, :-1], data_to_regress.iloc[:, -1]
        for fac in X.sum()[X.sum() == 0].index:
            if fac not in factors_x:
                del X[fac]
        w = X['ln_mkt_cap']

        _, cur_factor_ret, _ = regress(y, X, w) #重点: 回归出来是下期因子实际收益率
        cur_factor_ret.name = date              #实际内容是指date期的下一期因子实际收益率
        factor_rets = pd.concat([factor_rets, cur_factor_ret], axis=1)

    #因子实际收益率,行列转换一下,方便接下来处理
    factor_rets = factor_rets.T

    #对ROE_q以及growth因子的负值纠正为0
    for fac in ['ROE_q', 'growth']:
        try:
            fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
        except IndexError:
            continue
        factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] >= 0, 0)

    #利用历史[因子实际收益率]预测[因子预测收益率],当然预测方法可以有很多,这里只演示了两种
    if half_life:
        exp_wt = get_exponential_weights(window=window, half_life=half_life) #指数加权权重
        factor_rets = factor_rets.rolling(window=window).apply(wt_sum, args=(exp_wt,)).shift(1)
    else:
        #执行完下面代码后,factor_rets每个单元存放的内容是下一期[因子预测收益率].
        #预测方法: 假设当前T时刻,用过去window期的平均值作为T+1时刻预测值
        #代码解释: 一开始factor_rets每个单元存放的是对应日期的下一期[因子实际收益率],比如第12单元存放的是第13月的[因子实际收益率],
        #当执行完mean()这一步后,第12单元存放的是第13月的下一期(也就是第14月)的[因子预测收益率],然后需要执行shift(1)
        #将所有单元内容移动一格,那么 "第13月的下一期(就是第14月)的[因子预测收益率]" 就从第12单元移动到了第13单元,其他以此类推.
        factor_rets = factor_rets.rolling(window=window).mean().shift(1)
    factor_rets = factor_rets.dropna(how='all', axis=0)
    #重点1: 每个单元存放的都是相应下期因子预测收益率
    #重点2: [因子预测收益率序列]长度要比前面的[因子实际收益率序列]少window期
    return factor_rets

def get_est_stock_return(factors, factors_panel, est_factor_rets, window=12, half_life=6):
    """
    根据之前预测的[因子预测收益率],计算得到各股票的截面预期(预测)收益
    """
    est_stock_rets = pd.DataFrame()
    for date in est_factor_rets.index:
        cur_factor_panel = factors_panel[date] #date期因子值(因子暴露,因子载荷)
        cur_factor_panel = cur_factor_panel[factors]
        cur_factor_panel = cur_factor_panel.dropna(how='any', axis=0)
        cur_est_stock_rets = np.dot(cur_factor_panel, est_factor_rets.loc[date]) #参数: date期因子值, date+1期的[因子预测收益率]
        #实际内容是date+1期的股票预期收益率
        cur_est_stock_rets = pd.DataFrame(cur_est_stock_rets, index=cur_factor_panel.index, columns=[date])
        est_stock_rets = pd.concat([est_stock_rets, cur_est_stock_rets], axis=1)
    return est_stock_rets #重点:每个存储单元包含的是对应日期的下一期的股票预期收益率

def get_refresh_days(tradedays, start_date, end_date):
    """
    获取调仓日期（回测期内的每个月首个交易日）
    """
    tdays = tradedays
    sindex = get_date_idx(tradedays, start_date)
    eindex = get_date_idx(tradedays, end_date)
    tdays = tdays[sindex:eindex+1]
    return (nd for td, nd in zip(tdays[:-1], tdays[1:]) 
            if td.month != nd.month)

def get_date_idx(tradedays, date):
    """
    返回传入的交易日对应在全部交易日列表中的下标索引
    """
    datelist = list(tradedays)
    date = pd.to_datetime(date)
    try:
        idx = datelist.index(date)
    except ValueError:
        datelist.append(date)
        datelist.sort()
        idx = datelist.index(date)
        if idx == 0:
            return idx + 1
        else:
            return idx - 1
    return idx

def plot_net_value(records, benchmark, method_name, save_path, start_date, end_date):
    """
    绘制回测净值曲线
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    records = records[['benchmark_nv', 'net_value']]
    records /= records.iloc[0,:]
    plt.plot(records)
    plt.legend([benchmark, method_name], loc=2)
    plt.title('回测净值')
    plt.savefig(os.path.join(save_path, f'{method_name}_{start_date}-{end_date}.jpg'))
    plt.close()

def lp_solve(cur_est_rets, limit_factors, cur_benchmark_wt, num_multi=5):
    """
    线性规划计算函数：
    输入：截面预期收益，约束条件（风险因子），截面标的指数成分股权重，个股权重约束倍数
    输出：经优化后的组合内个股权重
    """
    data = pd.concat([cur_est_rets, limit_factors, cur_benchmark_wt], axis=1)
    data = data.dropna(how='any', axis=0)
    cur_est_rets, limit_factors, cur_benchmark_wt = (data.iloc[:, 0:1], data.iloc[:, 1:-1], data.iloc[:, -1])

    #****请勿改成简写形式（df /= df.sum())  错误原因待查****
    cur_benchmark_wt = cur_benchmark_wt / cur_benchmark_wt.sum()

    c = cur_est_rets.values.flatten()

    A_ub = None
    b_ub = None
    A_eq = np.r_[limit_factors.T.values, np.repeat(1, len(limit_factors)).reshape(1, -1)]
    b_eq = np.r_[np.dot(limit_factors.T, cur_benchmark_wt), np.array([1])]
    bounds = tuple([(0, num_multi * wt_in_index) for wt_in_index in cur_benchmark_wt.values])
    res = linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds)

    cur_wt = pd.Series(res.x, index=cur_est_rets.index)
    return cur_wt

def linear_programming(data_dict):
    """
    线性规划法-求解最优组合权重
    """
    est_stock_rets, limit_fac_data, index_wt = data_dict['est_stock_rets'], data_dict['limit_fac_data'], data_dict['index_wt']
    stock_wt = pd.DataFrame()
    for date in est_stock_rets.columns:
        est_rets = est_stock_rets[date]         #date+1期的股票预期收益率
        limit_fac_panel = limit_fac_data[date]  #date期的风险因子
        benchmark_wt = index_wt[date].dropna()  #date期的基准指数成分权重

        est_rets, limit_fac_panel = est_rets.loc[benchmark_wt.index], limit_fac_panel.loc[benchmark_wt.index]
        #求解date期股票权重向量,将来在date+1期股票开盘的时候,根据这个权重向量进行股票买卖,开仓,调仓等
        cur_wt = lp_solve(est_rets, limit_fac_panel, benchmark_wt)
        cur_wt.name = date
        stock_wt = pd.concat([stock_wt, cur_wt], axis=1)

    stock_wt = stock_wt.where(stock_wt != 0, np.nan)
    return stock_wt

def stratified_sample(data_dict):
    """
    分层抽样法-求解组合最优权重
    """
    data_panel = concat_factors_panel(None, data_dict, None, False, False)

    stock_wt = pd.DataFrame()
    for date in sorted(data_panel.keys()):
        panel = data_panel[date]
        if 'est_stock_rets' not in panel.columns:
            continue
        panel = panel.dropna(how='any', axis=0)
        panel_stkwt = pd.Series()
        for name, df in panel.groupby('industry_zx'):
            num = len(df) // 3
            remainder = len(df) % 3
            if len(df) <= 3:
                cur_ind_wt = df['index_wt']
                panel_stkwt = pd.concat([panel_stkwt, cur_ind_wt], axis=0)
            else:
                df = df.sort_values(by='MKT_CAP_FLOAT', ascending=False)
                if remainder == 1:
                    cut1, cut2 = num + 1, 2 * num + 1
                elif remainder == 2:
                    cut1, cut2 = num + 1, 2 * num + 2
                else:
                    cut1, cut2 = num, 2 * num
                df1, df2, df3 = df.iloc[:cut1, :], df.iloc[cut1:cut2, :], df.iloc[cut2:, :]
                for mkt_cap_group in [df1, df2, df3]:
                    max_code_idx = np.argmax(mkt_cap_group['est_stock_rets'])
                    cur_ind_wt = mkt_cap_group.loc[[max_code_idx], 'index_wt']
                    cur_ind_wt.loc[:] = np.sum(mkt_cap_group['index_wt'])
                    panel_stkwt = pd.concat([panel_stkwt, cur_ind_wt], axis=0)
        panel_stkwt.name = date
        stock_wt = pd.concat([stock_wt, panel_stkwt], axis=1)

    return stock_wt

def performance_attribution(factors_dict, index_wt, stock_wt, est_fac_rets, start_date, end_date):
    """
    业绩归因
    """
    factors_panel = concat_factors_panel(None, factors_dict, None, False, False)
    dates = stock_wt.loc[:, start_date:end_date].columns
    dates = pd.to_datetime(dates)
    res = pd.DataFrame()
    for date in dates:
        cur_index_wt = index_wt[date] / 100
        cur_index_wt = cur_index_wt / cur_index_wt.sum()
        w_delta = stock_wt[date] - cur_index_wt
        w_delta = w_delta.dropna()
        cur_factors_panel = factors_panel[date].loc[w_delta.index, :]
        cur_factor_exposure = w_delta.T @ cur_factors_panel
        cur_factor_exposure.name = date
        res = pd.concat([res, cur_factor_exposure], axis=1)

    res = res.T.groupby(pd.Grouper(freq='y')).mean()
    return res

def get_market_data(use_pctchg=True):
    global factor_generater
    if use_pctchg:
        market_data = factor_generater.pct_chg
    else:
        market_data = factor_generater.hfq_close
        market_data = market_data.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    market_data.columns = pd.to_datetime(market_data.columns)
    return market_data

def get_ori_name(factor_name, factors_to_concat):
    ''' 如果因子是合成因子或者是正交过的因子,取它们的原始名称
    '''
    if 'ortho' in factor_name: #先判断是否被正交过
        factor_name = factor_name[:-6]
    if 'con' in factor_name: #再判断是否合成因子
        pat = re.compile('(.*)_con_')
        ori_name = re.findall(pat, factor_name)[0]
        return factors_to_concat[ori_name]
    else:
        return [factor_name]

def factor_process(method, factors_to_concat, factors_ortho, index_wt, mut_codes, factors, risk_factors=None):
    #因子合成（等权）
    for factor_con, factors_to_con in factors_to_concat.items():
        factor_concat(factors_to_con, factor_con) #合成后的因子将保存到特定目录
    #因子正交
    for factor_x, factors_y in factors_ortho.items():
        orthogonalize(factors_y, factor_x, mut_codes, index_wt) #正交后的因子将保存到特定目录
    #开始打印相关系数热力图
    factors_to_corr = factors + risk_factors if risk_factors else factors
    factors_to_corr_ori = [name for fac in factors_to_corr 
                           for name in get_ori_name(fac, factors_to_concat)] #获取因子原始名称
    corr_ori = get_factor_corr(factors_to_corr_ori, mut_codes)
    plot_corr_heatmap(corr_ori, method, preprocessed=False) #因子合成或者正交前
    corr = get_factor_corr(factors_to_corr, mut_codes)
    plot_corr_heatmap(corr, method, preprocessed=True) #因子合成或者正交后
    print("相关系数热力图绘制完毕...")

def index_enhance_model(method='l', benchmark='000300.SH', start_date=None, end_date=None, methods=None):
    global index_enhance_dir
    lp_save_path = os.path.join(index_enhance_dir, '线性规划')
    ss_save_path = os.path.join(index_enhance_dir, '分层抽样')

    if method == 'l':
        method_name = 'linear_programming'
        save_path = lp_save_path
    elif method == 's':
        method_name = 'stratified_sample'
        save_path = ss_save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    wt_save_path = os.path.join(save_path, '优化权重')
    if not os.path.exists(wt_save_path):
        os.mkdir(wt_save_path)

    pctchgnm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    index_wt = get_stock_wt_in_index(benchmark) #基准指数权重
    mut_codes = index_wt.index.intersection(pctchgnm.index)

    data_dict = {} #核心权重优化函数的参数

    params = methods[method_name]
    factors, risk_factors, window, half_life = params['factors'], params['risk_factors'], params['window'], params['half_life']

    factors_dict = {fac: get_factor([fac], mut_codes)[fac] for fac in factors} #读取阿尔法因子的矩阵数据

    if method == 'l':
        risk_fac_data = {fac: get_factor([fac], mut_codes)[fac] for fac in risk_factors} #读取风险因子的矩阵数据
        limit_fac_data = concat_factors_panel(risk_factors, risk_fac_data, mut_codes, ind=True, mktcap=False) #矩阵转换成截面,同时加入行业伪变量
        data_dict.update({'limit_fac_data': limit_fac_data}) #优化函数的约束条件
    elif method == 's':
        risk_fac_data = {fac: get_factor([fac], mut_codes)[fac] for fac in ['industry_zx', 'MKT_CAP_FLOAT']}
        data_dict.update(risk_fac_data)

    #将alpha因子整理为截面形式
    factors_panel = concat_factors_panel(None, factors_dict, mut_codes)
    #利用
    est_fac_rets = factor_return_forecast(factors, factors_panel, window, half_life) #因子收益率预测
    est_fac_rets = est_fac_rets[factors]
    est_stock_rets = get_est_stock_return(factors, factors_panel, est_fac_rets, window, half_life) #计算股票预期收益率
    print('计算股票预期收益率完成...')

    mut_dates = index_wt.columns.intersection(est_stock_rets.columns)
    index_wt = index_wt.loc[mut_codes, mut_dates]
    est_stock_rets = est_stock_rets.loc[mut_codes, mut_dates]
    est_stock_rets.name = 'est_stock_return'

    data_dict.update({'index_wt': index_wt, 'est_stock_rets': est_stock_rets})
    #开始优化股票权重
    wt_cal_func = globals()[method_name]
    #重点:输入参数是t+1期(期末)股票预期收益率,输出结果是t期(期末)股票权重或者说是t+1期期初的股票权重
    stock_wt = wt_cal_func(data_dict)
    stock_wt = stock_wt / stock_wt.sum() #权重归一化
    print('计算股票权重完成...')

    #股票权重分析
    stock_wt.to_csv(os.path.join(wt_save_path, 'stock_wt.csv'), encoding='gbk')
    stock_weights_analysis(wt_save_path, stock_wt, index_wt)
    print('股票权重分析完成...')

    #接下来为回测做准备
    all_codes = stock_wt.index
    market_data = get_market_data(use_pctchg=False)
    benchmarkdata = market_data.loc[benchmark, start_date:end_date].T #基准指数日涨跌幅
    market_data = market_data.loc[all_codes, start_date:end_date] #基准指数所有成分股票的日涨跌幅
    #根据优化得到的各月末截面期HS300成分股股票权重,进行回测
    bt = Backtest_stock(market_data=market_data, 
                        start_date=start_date, 
                        end_date=end_date, 
                        benchmarkdata=benchmarkdata, 
                        stock_weights=stock_wt, 
                        use_pctchg=False)
    bt.run_backtest()
    print('回测结束, 进行回测结果分析...')
    summary_yearly = bt.summary_yearly() #回测统计
    summary_yearly.to_csv(os.path.join(save_path, f'回测统计_{start_date}至{end_date}.csv'), encoding='gbk')
    bt.portfolio_record.to_csv(os.path.join(save_path, f'回测净值_{start_date}至{end_date}.csv'), encoding='gbk')
    bt.position_record.to_csv(os.path.join(save_path, f'各期持仓_{start_date}至{end_date}.csv'), encoding='gbk')
    plot_net_value(bt.portfolio_record, benchmark, method_name, save_path, start_date, end_date)

    #业绩归因
    p_attr = performance_attribution(factors_dict, index_wt, stock_wt, est_fac_rets, start_date, end_date)
    p_attr.to_csv(os.path.join(save_path, f'业绩归因_{start_date}至{end_date}.csv'), encoding='gbk')
    print("分析结果存储完成!")

def plot_industry_bar(save_path, tdate, plot_data, title):
    plt.figure(figsize=(21, 15)) #it's a big plot.
    for i, (key, df) in zip(range(len(plot_data)), plot_data.items()):
        plt.subplot(int("12" + str(i+1))) #一行两列
        sns.barplot(x=df.index, y=key, data=df)
        plt.xticks(rotation=60) #rotate to avoid overlap text.
        plt.title(key, fontsize=21)
    plt.suptitle(title, fontsize=36)
    tdate = str(tdate)[:10]
    plt.savefig(f'{save_path}/{tdate}.png')
    plt.cla() #清除axes,即当前figure中的活动的axes,但其他axes保持不变
    plt.clf() #清除当前figure的所有axes,但是不关闭这个window,所以能继续复用于其他的plot
    plt.close('all') #关闭window,如果没有指定,则指当前window

def stock_weights_analysis(save_path, stock_wt, index_wt):
    global factor_generater

    wt_save_path = os.path.join(save_path, '行业股票权重')
    if not os.path.exists(wt_save_path):
        os.mkdir(wt_save_path)
    cn_save_path = os.path.join(save_path, '行业股票数量')
    if not os.path.exists(cn_save_path):
        os.mkdir(cn_save_path)
    hs_save_path = os.path.join(save_path, '每期持股详情')
    if not os.path.exists(hs_save_path):
        os.mkdir(hs_save_path)

    industry_citic = factor_generater.industry_citic

    stock_wt = stock_wt.shift(1, axis=1) #每个单元存放的是下一期的权重
    stock_wt = stock_wt.iloc[:, 1:]

    index_wt = index_wt / index_wt.sum() #除以100其实更合适

    for tdate in stock_wt.columns:
        caldate = factor_generater.month_map[tdate]
        #打印行业股票权重分布图
        df_stock_wt = pd.DataFrame(stock_wt[tdate].values, index=stock_wt[tdate].index, columns=['stock_wt'])
        df_stock_wt = df_stock_wt.dropna()
        df_stock_wt['industry'] = industry_citic.loc[df_stock_wt.index, caldate]
        df_index_wt = pd.DataFrame(index_wt[tdate].values, index=index_wt[tdate].index, columns=['index_wt'])
        df_index_wt = df_index_wt.dropna()
        df_index_wt['industry'] = industry_citic.loc[df_index_wt.index, caldate]
        plot_data = {}
        plot_data['stock_wt'] = df_stock_wt.groupby(by='industry').sum()
        plot_data['index_wt'] = df_index_wt.groupby(by='industry').sum()
        plot_industry_bar(wt_save_path, tdate, plot_data, "选股权重与基准指数(沪深300)行业比较")
        #打印行业股票数量分布图
        df_stock_count = pd.DataFrame(stock_wt[tdate].values, index=stock_wt[tdate].index, columns=['stock_count'])
        df_stock_count['industry'] = industry_citic.loc[df_stock_count.index, caldate]
        df_index_count = pd.DataFrame(index_wt[tdate].values, index=index_wt[tdate].index, columns=['index_count'])
        df_index_count['industry'] = industry_citic.loc[df_index_count.index, caldate]
        plot_data = {}
        plot_data['stock_count'] = df_stock_count.groupby(by='industry').count()
        plot_data['index_count'] = df_index_count.groupby(by='industry').count()
        plot_industry_bar(cn_save_path, tdate, plot_data, "选股数量与基准指数(沪深300)行业比较")
        #保存每期持股详情
        df_stock_wt['sec_name'] = factor_generater.meta.loc[df_stock_wt.index, 'sec_name']
        df_stock_wt['mkt_cap_float'] = factor_generater.mkt_cap_float.loc[df_stock_wt.index, tdate]
        df_stock_wt['turn'] = factor_generater.turn.loc[df_stock_wt.index, tdate]
        df_stock_wt['amt'] = factor_generater.amt.loc[df_stock_wt.index, tdate]
        df_stock_wt['close'] = factor_generater.close.loc[df_stock_wt.index, tdate]
        name = str(tdate)[:10]
        df_stock_wt.to_csv(os.path.join(hs_save_path, f'{name}.csv'), encoding='gbk')