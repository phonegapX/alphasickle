# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os
import warnings
from single_factor_test import *

warnings.filterwarnings('ignore')  #将运行中的警告信息设置为“忽略”，从而不在控制台显示

#工作目录，存放代码和因子基本信息
work_dir = os.path.dirname(os.path.dirname(__file__))
#经过预处理后的因子截面数据存放目录
factor_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
#测试结果图表存放目录
sf_test_save_path = os.path.join(work_dir, '单因子检验')

def main():
    factors = input("请输入待进行检验的因子（以,分隔），'a'为全部因子：")
    if factors == 'a':
        factors = get_factor_names()
    else:
        factors = factors.split(',')

    single_factor_test(factors)
    layer_division_backtest(factors)

if __name__ == '__main__':
    main()
