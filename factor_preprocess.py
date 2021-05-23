# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from functools import reduce
from single_factor_test import panel_to_matrix

sns.set(style="darkgrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    #正常显示负号
plt.rcParams['figure.figsize'] = (16.0, 9.0)  #图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 15                #字体大小

#基准信息所在列名（分别对应：
#code - 证券wind代码；name - 证券简称；ipo_date - 上市日期；
#industry_zx - 中信一级行业；industry_sw - 申万一级行业；
#MKT_CAP_FLOAT - 流通市值；is_open1 - 当日是否开盘；
#PCT_CHG_NM - 下个月的月收益率
info_cols = ['code', 'name', 'ipo_date', 'industry_zx', 'industry_sw', 'MKT_CAP_FLOAT', 'is_open1', 'PCT_CHG_NM']

work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '因子预处理模块')
file_path = os.path.join(work_dir, '因子（原始）') #原始因子数据所在目录
save_path = os.path.join(work_dir, '因子（已预处理）') #预处理后因子数据保存目录
visu_path = os.path.join(work_dir, '因子可视化')
quality_path = os.path.join(work_dir, '因子数据品质检验')
matrix_path = os.path.join(work_dir, '矩阵数据')
inscmp_path = os.path.join(work_dir, '因子行业比较')

industry_benchmark = 'zx' #行业基准（用于缺失值填充和中性化）
                          #zx - 中信； sw - 申万

def get_factor_data(datdf, names=None):
    """
    根据输入的因子名称将原始因子截面数据分割
    """
    global info_cols
    if names:
        try:
            fac_names = [fac.lower() for fac in datdf.columns]
            idx = [fac_names.index(fac_name.lower()) for fac_name in names] #idx理解为列位置索引,为避免大小写的判断
        except:
            msg = "请重新确认因子名称是否正确"
            raise Exception(msg)

    #对截面数据中，基准信息列存在空缺的股票（行）进行删除处理
    cond = ~pd.isnull(datdf['MKT_CAP_FLOAT']) #给cond赋初值
    for col in info_cols:
        if col == 'PCT_CHG_NM':
            if pd.isnull(datdf['PCT_CHG_NM']).all(): #比如最后一个截面的数据就会全为null
                continue
        if col != 'MKT_CAP_FLOAT':
            cond &= ~pd.isnull(datdf[col]) #更新cond
    datdf = datdf.loc[cond]

    #将原截面数据按照预处理与否分别划分，返回需处理和不需处理2个因子截面数据
    if names is None: #全部处理
        return datdf, pd.DataFrame()
    else:
        dat_to_process = datdf.iloc[:, idx] #用户选择的因子
        dat_to_process = pd.merge(datdf[info_cols], dat_to_process, left_index=True, right_index=True)
        unchanged_cols = sorted(set(datdf.columns) - set(dat_to_process.columns))
        dat_unchanged = datdf[unchanged_cols]
    return dat_to_process, dat_unchanged

def fill_na(data, ind='zx'):
    """
    缺失值填充：缺失值少于10%的情况下使用行业中位数代替
    本函数执行完以后如果仍然有因子存在缺失值,一般两种情况:
    1.因子缺失值超过10% (缺失值超过10%的因子不具备统计意义,没法进行选股)
    2.某些因子在某些行业没有取值 (比如cashratio在银行行业就没有取值,这样的因子没法进行全行业选股)
    """
    global info_cols
    datdf = data.copy()

    facs_to_fill = datdf.columns.difference(set(info_cols)) #排除掉一些基准列
    facs_to_fill = [fac for fac in facs_to_fill             #筛选缺失值少于10%的因子
                    if pd.isnull(datdf[fac]).sum() / len(datdf) <= 0.3] #True为1,False为0

    for fac in facs_to_fill:
        try:
            fac_median_by_ind = datdf[[f'industry_{ind}', fac]].groupby(f'industry_{ind}').median()
        except:
            #强制转换为数字类型
            datdf.loc[:, fac] = datdf[[fac]].applymap(coerce_numeric)
            fac_median_by_ind = datdf[[f'industry_{ind}', fac]].groupby(f'industry_{ind}').median()
        fac_ind_map = fac_median_by_ind.to_dict()[fac]
        fac_to_fill = datdf.loc[pd.isnull(datdf[fac]), [f'industry_{ind}', fac]] #找出所有为空的因子字段
        fac_to_fill.loc[:, fac] = fac_to_fill[f'industry_{ind}'].map(fac_ind_map) #填充这些为空的因子字段
        datdf.loc[fac_to_fill.index, fac] = fac_to_fill[fac].values

    #如果仍然有因子存在缺失值,一般两种情况:
    #1.因子缺失值超过10% (缺失值超过10%的因子不具备统计意义,没法进行选股)
    #2.某些因子在某些行业没有取值 (比如cashratio在银行行业就没有取值,这样的因子没法进行全行业选股)
    #这两种情况下的因子都不能使用,所以把这样的因子全部设置为空值
    facs_to_fill = datdf.columns.difference(set(info_cols)) #排除掉一些基准列
    facs_to_fill = [fac for fac in facs_to_fill if pd.isnull(datdf[fac]).sum()>0] #True为1,False为0
    datdf[facs_to_fill] = np.nan

    return datdf

def coerce_numeric(s):
    try:
        return float(s)
    except:
        return -np.inf

def winsorize(data, n=5):
    """
    去极值：5倍中位数标准差法（5mad）
    """
    global info_cols
    datdf = data.copy()
    #存在空值的因子列不处理,直接忽略
    if_contain_na = pd.isnull(datdf).sum().sort_values(ascending=True)
    facs_to_remove = if_contain_na.loc[if_contain_na > 0].index.tolist()
    if 'PCT_CHG_NM' in facs_to_remove: #PCT_CHG_NM是允许为空的,所以不应该被删除,从删除列表里面剔除
        facs_to_remove.remove('PCT_CHG_NM')
    facs_to_win = datdf.columns.difference(set(info_cols)).difference(set(tuple(facs_to_remove))) #去掉基准列和要被删除的因子列
    dat_win = datdf[facs_to_win]
    fac_vals = dat_win.values
    dm = np.nanmedian(fac_vals, axis=0)
    dm1 = np.nanmedian(np.abs(fac_vals - dm), axis=0)
    dm = np.repeat(dm.reshape(1,-1), fac_vals.shape[0], axis=0)
    dm1 = np.repeat(dm1.reshape(1,-1), fac_vals.shape[0], axis=0)
    fac_vals = np.where(fac_vals > dm + n*dm1, dm+n*dm1, np.where(fac_vals < dm - n*dm1, dm - n*dm1, fac_vals))
    res = pd.DataFrame(fac_vals, index=dat_win.index, columns=dat_win.columns)
    datdf.loc[res.index, facs_to_win] = res
    return datdf

def neutralize(data, ind='zx'):
    """
    中性化：因子暴露度对行业哑变量（ind_dummy_matrix）和对数流通市值（lncap_barra）
            做线性回归, 取残差作为新的因子暴露度
    """
    global info_cols
    datdf = data.copy()
    cols_to_neu = datdf.columns.difference(set(info_cols))
    y = datdf[cols_to_neu]
    y = y.dropna(how='any', axis=1) #存在空值的因子就不处理
    cols_neu = y.columns
    lncap = np.log(datdf[['MKT_CAP_FLOAT']])
    ind_dummy_matrix = pd.get_dummies(datdf[f'industry_{ind}'])
    X = pd.concat([lncap, ind_dummy_matrix], axis=1)
    model = LinearRegression(fit_intercept=False)
    res = model.fit(X, y)
    coef = res.coef_
    residue = y - np.dot(X, coef.T)
    assert len(datdf.index.difference(residue.index)) == 0
    datdf.loc[residue.index, cols_neu] = residue
    return datdf

def standardize(data):
    """
    标准化：Z-score标准化方法，减去均值，除以标准差
    """
    global info_cols
    datdf = data.copy()
    facs_to_sta = datdf.columns.difference(set(info_cols))
    dat_sta = datdf[facs_to_sta].values
    dat_sta = (dat_sta - np.mean(dat_sta, axis=0)) / np.std(dat_sta, axis=0)
    datdf.loc[:, facs_to_sta] = dat_sta
    return datdf

def process_input_names(factor_names):
    if factor_names == 'a':
        factor_names = None
    else:
        factor_names = [f.replace("'","").replace('"',"") for f in factor_names.split(',')]
    return factor_names

def plot_factor_data(show_path, sub_type, name, datdf):
    global info_cols
    save_path = os.path.join(show_path, sub_type)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    datdf = datdf.dropna(how='all', axis=1)
    facs = datdf.columns.difference(set(info_cols))
    facs = facs[:81] #只查看前81个
    plt.figure(figsize=(32, 36))
    for i, factor in zip(range(len(facs)), facs):
        plt.subplot(9,9,i+1)
        sns.distplot(datdf[factor].fillna(0).values)
        plt.title(factor)
    plt.suptitle(f'{sub_type}-{name}')
    plt.savefig(f'{save_path}\\{name}.png')
    #plt.cla() #清除axes,即当前figure中的活动的axes,但其他axes保持不变
    #plt.clf() #清除当前figure的所有axes,但是不关闭这个window,所以能继续复用于其他的plot
    plt.close('all') #关闭window,如果没有指定,则指当前window

def factor_data_quality_check(factors_path, factor_names, save_path, sub_dir_name, usable_factor_stat=False):
    save_path = os.path.join(save_path, sub_dir_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df_null_stat = None
    df_zero_stat = None
    l = os.listdir(factors_path)[:]
    index = [pd.to_datetime(f[:10]) for f in l]
    l_usable_factor_stat = []
    for f in l:
        dt = pd.to_datetime(f[:10])
        data = pd.read_csv(os.path.join(factors_path, f), engine='python', encoding='gbk', index_col=[0])
        datdf, _ = get_factor_data(data, factor_names)
        facs = datdf.columns.difference(set(info_cols)) #排除掉一些基准列
        datdf = datdf[facs]
        null_stat = pd.isnull(datdf).sum() / len(datdf) #True为1,False为0
        zero_stat = (datdf==0).sum() / len(datdf) #True为1,False为0
        if (df_null_stat is None):
            df_null_stat = pd.DataFrame(index=index, columns=facs)
        if (df_zero_stat is None):
            df_zero_stat = pd.DataFrame(index=index, columns=facs)
        df_null_stat.loc[dt] = null_stat
        df_zero_stat.loc[dt] = zero_stat
        #
        if usable_factor_stat:
            #同样的因子在有些截面期可能会为空,对于我们模型来说一个因子只有在所有截面期所有股票上都有取值的情况下才有意义
            #将所有截面期上所有股票都有取值的因子统计出来,这些因子才能用于接下来的多因子模型
            l_usable_factor_stat.append(pd.isnull(datdf).sum()>0)
        print(dt)
    df_null_stat.to_csv(os.path.join(save_path, "缺失值统计.csv"), encoding='gbk')
    df_zero_stat.to_csv(os.path.join(save_path, "零值统计.csv"), encoding='gbk')
    #
    if usable_factor_stat:
        r = reduce(lambda left,right:left|right, l_usable_factor_stat)
        r = r[~r]
        r[:] = np.nan
        r.to_csv(os.path.join(save_path, "可用因子统计.csv"), encoding='gbk')

def process_cross_section(fpath, factor_names=None, visualized=False):
    """
    输入： 需要进行预处理的因子名称（可为1个或多个，默认为对所有因子进行预处理）
    输出： 预处理后的因子截面数据（如2009-01-23.csv文件）
    
    对指定的原始因子数据进行预处理
    顺序：缺失值填充、去极值、中性化、标准化
    （因输入的截面数据中所含财务类因子默认已经
    财务日期对齐处理，故在此不再进行该步处理）
    """
    global file_path, save_path, visu_path, industry_benchmark
    #读取原始因子截面数据
    data = pd.read_csv(os.path.join(file_path, fpath), engine='python', encoding='gbk', index_col=[0])
    #根据输入的因子名称将原始因子截面数据分割
    data_to_process, data_unchanged = get_factor_data(data, factor_names)
    #预处理步骤依次进行
    name = fpath[:10]
    if visualized: plot_factor_data(visu_path, "原始值", name, data_to_process)
    data_to_process = fill_na(data_to_process, industry_benchmark)      #缺失值填充
    if visualized: plot_factor_data(visu_path, "缺失值填充后", name, data_to_process)
    data_to_process = winsorize(data_to_process)                        #去极值
    if visualized: plot_factor_data(visu_path, "去极值后", name, data_to_process)
    data_to_process = neutralize(data_to_process, industry_benchmark)   #中性化
    if visualized: plot_factor_data(visu_path, "中性化后", name, data_to_process)
    data_to_process = standardize(data_to_process)                      #标准化
    if visualized: plot_factor_data(visu_path, "标准化后", name, data_to_process)
    #合并生成经过处理后的总因子文件
    if len(data_unchanged) > 0:
        data_final = pd.concat([data_to_process, data_unchanged.loc[data_to_process.index]], axis=1)
    else:
        data_final = data_to_process
    data_final.index = range(1, len(data_final)+1)
    data_final.index.name = 'No'
    data_final.to_csv(os.path.join(save_path, fpath), encoding='gbk')
    print(f'{name}处理完毕')

def plot_industry_comparison(factor_name, plot_data):
    global inscmp_path    
    plt.figure(figsize=(15, 12))
    plt.subplot(111)
    sns.barplot(x=plot_data.index, y=factor_name, data=plot_data)
    plt.xticks(rotation=60) #rotate to avoid overlap text.
    plt.title(factor_name, fontsize=21)
    plt.suptitle('因子行业比较', fontsize=36)
    plt.savefig(f'{inscmp_path}/{factor_name}.png')
    plt.cla() #清除axes,即当前figure中的活动的axes,但其他axes保持不变
    plt.clf() #清除当前figure的所有axes,但是不关闭这个window,所以能继续复用于其他的plot
    plt.close('all') #关闭window,如果没有指定,则指当前window

def factor_industry_comparison(panel_path, matrix_path, sub_dir_name):
    #截面格式转换成矩阵格式
    matrix_path = os.path.join(matrix_path, sub_dir_name)
    if not os.path.exists(matrix_path):
        os.mkdir(matrix_path)
    matrix_path = os.path.join(matrix_path, "因子矩阵")
    if not os.path.exists(matrix_path):
        os.mkdir(matrix_path)
    f = (os.listdir(panel_path)[:])[0]
    data = pd.read_csv(os.path.join(panel_path, f), engine='python', encoding='gbk', index_col=[0])
    datdf, _ = get_factor_data(data, None)
    facs = datdf.columns.difference(set(info_cols)) #排除掉一些基准列
    panel_to_matrix(facs, factor_path=panel_path, save_path=matrix_path)
    #读取行业分类
    industry_citic = pd.read_csv(os.path.join(matrix_path, 'industry_zx.csv'), engine='python', encoding='gbk', index_col=[0])
    industry_citic.columns = pd.to_datetime(industry_citic.columns)
    for fac in facs:
        fac = fac.replace('/', '_div_')
        r_df = pd.DataFrame()
        datdf = pd.read_csv(os.path.join(matrix_path, f'{fac}.csv'), engine='python', encoding='gbk', index_col=[0])
        datdf.columns = pd.to_datetime(datdf.columns)
        for dt in datdf.columns: #按期统计,因为股票不同期行业可能会有变化,虽然发生的概率很小
            df = pd.DataFrame()
            df['factor'] = datdf[dt]
            df['industry'] = industry_citic[dt]
            df = df.dropna()
            r_df[dt] = df.groupby(by='industry').mean()['factor']
        r_df = r_df.mean(axis=1).to_frame(fac)
        plot_industry_comparison(fac, r_df)
        print(fac)