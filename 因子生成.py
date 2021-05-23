# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os
import pandas as pd
from joblib import Parallel, delayed
from factor_generate import FactorGenerater, FileAlreadyExistError, WORK_PATH

gen = FactorGenerater()

def create_factor_file(date):
    sname = str(date)[:10]
    try:
        gen.create_factor_file(date, os.path.join(WORK_PATH, "factors", f"{sname}.csv"))
    except FileAlreadyExistError:
        print(f"{sname}'s data already exists.")
    else:
        print(f"Create {sname}'s data complete.")

def main():
    dates = [d for d in gen.month_map.keys()]
    s = pd.to_datetime('20120101')
    e = pd.to_datetime('20191130')
    dates = pd.Series(dates, index=dates)
    dates = dates[(dates>=s)&(dates<=e)]
    #串行
    #for date in dates:
    #    create_factor_file(date)
    #并行
    function_list = [delayed(create_factor_file)(date) for date in dates]
    Parallel(n_jobs=5, backend='multiprocessing')(function_list) #并行化处理

if __name__ == '__main__':
    main()