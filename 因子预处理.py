# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os
from joblib import Parallel, delayed
from factor_preprocess import *

def input_yes_or_no(tips):
    while True:
        r = input(tips)
        if r in ['y','Y']: visualized = True; break
        elif r in ['n','N']: visualized = False; break
        else: continue    
    return visualized

def main():
    #创建处理后因子的存放目录
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #
    if not os.path.exists(visu_path):
        os.mkdir(visu_path)
    #
    if not os.path.exists(quality_path):
        os.mkdir(quality_path)
    #
    if not os.path.exists(matrix_path):
        os.mkdir(matrix_path)
    #
    if not os.path.exists(inscmp_path):
        os.mkdir(inscmp_path)
    #
    print('开始进行因子(原始)数据品质检验(缺失值和零值统计)...')
    factor_data_quality_check(file_path, None, quality_path, "原始")
    input(f'检验完毕,请到 {quality_path} 查看结果. 按任意键继续...')
    #
    is_cmp = input_yes_or_no("是否进行因子行业比较(y/n)?: ")
    if is_cmp:
        factor_industry_comparison(file_path, matrix_path, "原始")
        input(f'比较完毕,请到 {inscmp_path} 查看结果. 按任意键继续...')
    #收集需要处理的因子名称
    while True:
        factor_names = input("请输入需预处理的因子名称（请使用英文逗号','分隔多个因子名称，输入'a'代表全部处理）：")
        if factor_names: break
    factor_names = process_input_names(factor_names)
    #对所有横截面数据进行遍历（2009-01至2019-01每个月月末（交易日））
    visualized = input_yes_or_no("是否进行因子可视化操作(y/n)?启用会更加耗时: ")
    #串行
    #for fpath in os.listdir(file_path)[:]:
    #    process_cross_section(fpath, factor_names, visualized)
    #并行
    function_list = [delayed(process_cross_section)(fpath, factor_names, visualized) for fpath in os.listdir(file_path)[:]]
    Parallel(n_jobs=10, backend='multiprocessing')(function_list) #并行化处理
    print('因子截面数据已全部处理！')
    #
    print('开始进行因子(已预处理)数据品质检验(缺失值和零值统计)...')
    factor_data_quality_check(save_path, factor_names, quality_path, "已预处理", True)
    print(f'检验完毕,请到 {quality_path} 查看结果.')

if __name__ == '__main__':
    main()
