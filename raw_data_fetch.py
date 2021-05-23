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
import tushare as ts
import pymysql
from retrying import retry
from functools import wraps
from factor_generate import FactorGenerater
try:
    basestring
except NameError:
    basestring = str

#打印能完整显示
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 50000)
pd.set_option('max_colwidth', 1000)


class RawDataFetcher(FactorGenerater):

    def _get_month_end(self, date):
        import calendar
        import pandas.tseries.offsets as toffsets
        _, days = calendar.monthrange(date.year, date.month)
        if date.day == days:
            return date
        else:
            return date + toffsets.MonthEnd(n=1)

    @retry(stop_max_attempt_number=500, wait_random_min=1000, wait_random_max=2000)
    def ensure_data(self, func, save_dir, start_dt='20010101', end_dt='20201231'):
        """ 确保按交易日获取数据
        """
        tmp_dir = os.path.join(self.root, save_dir)
        dl = [pd.to_datetime(name.split(".")[0]) for name in os.listdir(tmp_dir)]
        dl = sorted(dl)
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        tdays = pd.Series(self.tradedays, index=self.tradedays)
        tdays = tdays[(tdays>=s)&(tdays<=e)]
        tdays = tdays.index.tolist()
        for tday in tdays:
            if tday in dl: continue
            t = tday.strftime("%Y%m%d")
            datdf = func(t)
            path = os.path.join(tmp_dir, t+".csv")
            datdf.to_csv(path, encoding='gbk')
            print(t+".csv write ok !!!!!")

    @retry(stop_max_attempt_number=500, wait_random_min=1000, wait_random_max=2000)
    def ensure_data_by_q(self, func, save_dir, start_dt='20010101', end_dt='20201231'):
        """ 确保按季度获取数据
        """
        tmp_dir = os.path.join(self.root, save_dir)
        dl = [pd.to_datetime(name.split(".")[0]) for name in os.listdir(tmp_dir)]
        dl = sorted(dl)
        if len(dl) > 3:
            dl = dl[0:len(dl)-3] #已经存在的最后三个季度数据重新下载
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        qdates = pd.date_range(start=s, end=e, freq='Q')
        qdates = qdates.tolist()
        for tday in qdates:
            if tday in dl: continue
            t = tday.strftime("%Y%m%d")
            datdf = func(period=t)
            path = os.path.join(tmp_dir, t+".csv")
            datdf.to_csv(path, encoding='gbk')
            print(t+".csv write ok !!!!!")

    def create_indicator(self, raw_data_dir, raw_data_field, indicator_name):
        ''' 主要用于通过日频数据创建日频指标
        '''
        tmp_dir = os.path.join(self.root, raw_data_dir)
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays)
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col=['ts_code'], engine='python', encoding='gbk')
            df[tday] = dat[raw_data_field]
            print(tday)
        df = df.dropna(how='all') #删掉全为空的一行
        diff = df.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
        df = df.drop(labels=diff)
        self.close_file(df, indicator_name)

    def create_indicator_m_by_d(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20010101', end_dt='20201231'):
        ''' 通过日频数据创建月频指标
        '''
        tmp_dir = os.path.join(self.root, raw_data_dir)
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat[raw_data_field]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    def create_indicator_m_by_d_ex(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20010101', end_dt='20201231'):
        ''' 通过日频数据创建月频指标
        '''
        self.create_indicator(raw_data_dir, raw_data_field, indicator_name)
        datdf = getattr(self, indicator_name, None)
        datdf = self.preprocess(datdf)
        self.close_file(datdf, indicator_name)
        #
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            caldate = self.month_map[tday]
            df[caldate] = datdf[tday]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name+'_m')

    def create_indicator_m_by_q(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20010101', end_dt='20201231'):
        ''' 通过季频数据创建月频指标,主要用于财报数据处理
        '''
        s = pd.to_datetime(start_dt) #统计周期开始
        e = pd.to_datetime(end_dt) #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天
        all_stocks_info = self.meta
        tmp_dir = os.path.join(self.root, raw_data_dir) #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        datpanel = pd.Panel(panel)
        datpanel = datpanel.to_frame().stack().unstack(level=(0,1)) #貌似某些情况下会有BUG,有索引但是没数据
        #开始计算结果指标(月频),在每个时间截面逐个处理每只股票
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            for stock in df.index: #每只股票
                try:
                    datdf = datpanel[stock]
                    datdf = datdf.loc[datdf['ann_date']<d] #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
                    df.at[stock, d] = datdf.iloc[-1].at[raw_data_field] #取已经发布最近一期财报数据指定字段进行赋值
                    #print(stock)
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    def create_indicator_m_by_q_ex(self, raw_data_dir, raw_data_field, indicator_name, start_dt='20010101', end_dt='20201231'):
        ''' 通过季频数据创建月频指标,主要用于财报数据处理
        '''
        s = pd.to_datetime(start_dt) #统计周期开始
        e = pd.to_datetime(end_dt) #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天
        all_stocks_info = self.meta
        tmp_dir = os.path.join(self.root, raw_data_dir) #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=['ts_code'], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        datpanel = pd.Panel(panel)
        datpanel = datpanel.swapaxes(0, 1)
        #开始计算结果指标(月频),在每个时间截面逐个处理每只股票
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            for stock in df.index: #每只股票
                try:
                    datdf = datpanel.loc[stock]
                    datdf = datdf.loc[datdf['ann_date']<d] #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
                    df.at[stock, d] = datdf.iloc[-1].at[raw_data_field] #取已经发布最近一期财报数据指定字段进行赋值
                    #print(stock)
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, indicator_name)

    def _align_element(self, df1, df2):
        ''' 对齐股票和时间
        '''
        row_index = sorted(df1.index.intersection(df2.index))
        col_index = sorted(df1.columns.intersection(df2.columns))
        return df1.loc[row_index, col_index], df2.loc[row_index, col_index]

    def create_daily_quote_indicators(self):
        '''
        '''
        #-------------------------------------------------------------
        #创建一些行情指标
        self.create_indicator("__temp_daily__", "S_DQ_ADJFACTOR", "adjfactor")
        adjfactor = self.preprocess(self.adjfactor)
        self.close_file(adjfactor, 'adjfactor')

        self.create_indicator("__temp_daily__", "amount", "amt")
        amt = self.amt / 10 #默认每单位千元,转换为每单位万元
        amt = self.preprocess(amt, suspend_days_process=True, val=0)
        self.close_file(amt, 'amt')

        self.create_indicator("__temp_daily__", "close", "close")
        close = self.preprocess(self.close)
        self.close_file(close, 'close')

        close, adjfactor = self._align_element(self.close, self.adjfactor)
        hfq_close = close * adjfactor
        self.close_file(hfq_close, 'hfq_close') #后复权收盘价

        self.create_indicator("__temp_daily__", "pct_chg", "pct_chg")
        pct_chg = self.preprocess(self.pct_chg, suspend_days_process=True, val=0)
        self.close_file(pct_chg, 'pct_chg')
        #-------------------------------------------------------------
        #将三大指数的数据给补上
        pct_chg = self.pct_chg
        close = self.close
        hfq_close = self.hfq_close
        benchmarks = ['000001.SH', '000300.SH', '000905.SH'] #上证综指,沪深300,中证500
        tmp_dir = os.path.join(self.root, "__temp_index_daily__")
        for name in benchmarks:
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[2], engine='python', encoding='gbk', parse_dates=['trade_date'])
            pct_chg.loc[name] = dat['pct_chg'][pct_chg.columns]
            close.loc[name] = dat['close'][close.columns]
            hfq_close.loc[name] = dat['close'][hfq_close.columns]
        #更新数据
        pct_chg = pct_chg / 100
        self.close_file(pct_chg, 'pct_chg')
        self.close_file(close, 'close')
        self.close_file(hfq_close, 'hfq_close')
        #-------------------------------------------------------------
        #生成周期为1,3,6,12月收益率
        s = pd.to_datetime('20010101') #统计周期开始
        e = pd.to_datetime('20201231') #统计周期结束
        tdays_be_month = self.trade_days_begin_end_of_month
        tdays_be_month = tdays_be_month[(tdays_be_month>=s)&(tdays_be_month<=e)].dropna(how='all')
        months_end = tdays_be_month.index
        hfq_close = self.hfq_close
        #***pct_chg_M
        pct_chg_M = pd.DataFrame()
        for m_end_date in months_end:
            m_start_date = tdays_be_month.loc[m_end_date].values[0]
            pct_chg_M[self.month_map.loc[m_end_date]] = hfq_close[m_end_date] / hfq_close[m_start_date] - 1
        self.close_file(pct_chg_M, 'pct_chg_M')
        #pct_chg_Nm
        for period in (1,3,6,12):
            pct_chg_Nm = pd.DataFrame()
            if period != 1: 
                for m_end_date in months_end[::-1]:
                    try:
                        start_date_before_n_period = tdays_be_month.loc[self._get_date(m_end_date, -period+1, months_end)].values[0]
                        s = hfq_close[m_end_date] / hfq_close[start_date_before_n_period] - 1
                        pct_chg_Nm[self.month_map[m_end_date]] = s
                    except KeyError:
                        print(m_end_date)
                        break
            else:
                pct_chg_Nm = getattr(self, f'pct_chg_M', None)
            self.close_file(pct_chg_Nm, f"pctchg_{period}M")
            print(f'pct_chg_{period}M updated.')

    def create_daily_basic_indicators(self):
        '''
        '''
        self.create_indicator("__temp_daily_basic__", "turnover_rate", "turn")
        turn = self.turn / 100
        turn = self.preprocess(turn, suspend_days_process=True)
        self.close_file(turn, "turn")

        self.create_indicator("__temp_daily_basic__", "total_mv", "mkt_cap_ard")
        mkt_cap_ard = self.preprocess(self.mkt_cap_ard)
        self.close_file(mkt_cap_ard, "mkt_cap_ard")

    def preprocess(self, datdf, suspend_days_process=False, val=np.nan):
        ''' 数据预处理
        '''
        datdf = datdf.copy()
        datdf = datdf.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        row_index, col_index = datdf.index, datdf.columns
        liststatus = self.listday_matrix.loc[row_index, col_index]
        cond = (liststatus==1)
        datdf = datdf.where(cond) #将不是上市日的数值替换为nan
        if suspend_days_process:
            tradestatus = self.trade_status.loc[row_index, col_index]
            cond = (liststatus==1) & (tradestatus==0)
            datdf = datdf.where(~cond, val) #将上市但停牌的数值设为指定值
        return datdf


class TushareFetcher(RawDataFetcher):

    def __init__(self):
        self.pro = ts.pro_api('x')
        super().__init__(using_fetch=True)

    def fetch_meta_data(self):
        """ 股票基础信息
        """
        df_list = []
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date')
        df_list.append(df)
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date', list_status='D')
        df_list.append(df)
        df = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date', list_status='P')
        df_list.append(df)
        df = pd.concat(df_list)
        df = df.rename(columns={"list_date":"ipo_date"})
        df = df.rename(columns={'name':'sec_name'})
        df = df.rename(columns={"ts_code":"code"})
        df.drop_duplicates(subset=['code'], keep='first', inplace=True)
        df.sort_values(by=['ipo_date'], inplace=True)
        #print(pd.to_datetime(df['ipo_date']))
        #df.reset_index(drop=True, inplace=True)
        df.set_index(['code'], inplace=True)
        self.close_file(df, 'meta')

    def fetch_trade_day(self):
        """ 交易日列表
        """
        df = self.pro.trade_cal(is_open='1')
        df = df[['cal_date','is_open']]
        df = df.rename(columns={"cal_date":"tradedays"})
        df.set_index(['tradedays'], inplace=True)
        self.close_file(df, 'tradedays')

    def fetch_month_map(self):
        """ 每月最后一个交易日和每月最后一个日历日的映射表
        """
        tdays = self.tradedays
        s_dates = pd.Series(tdays, index=tdays)
        func_last = lambda ser: ser.iat[-1]
        new_dates = s_dates.resample('M').apply(func_last)
        month_map = new_dates.to_frame(name='trade_date')
        month_map.index.name = 'calendar_date'
        month_map.reset_index(inplace=True)
        month_map.set_index(['trade_date'], inplace=True)
        self.close_file(month_map, 'month_map')

    #------------------------------------------------------------------------------------
    #日数据
    def daily(self, t):
        return self.pro.daily(trade_date=t)

    def suspend_d(self, t):
        return self.pro.suspend_d(trade_date=t)

    def limit_list(self, t):
        return self.pro.limit_list(trade_date=t)

    def adj_factor(self, t):
        return self.pro.adj_factor(trade_date=t)

    def daily_basic(self, t):
        return self.pro.daily_basic(trade_date=t)

    def moneyflow(self, t):
        return self.pro.moneyflow(trade_date=t)
    #------------------------------------------------------------------------------------

    def segment_op(limit, _max):
        """ 分段获取数据
        """
        def segment_op_(f):
            #
            @wraps(f)
            def wrapper(*args, **kwargs):
                dfs = []
                for i in range(0, _max, limit):
                    kwargs['offset'] = i
                    df = f(*args, **kwargs)
                    if len(df) < limit:
                        if len(df) > 0:
                            dfs.append(df)
                        break
                    df = df.iloc[0:limit]
                    dfs.append(df)
                df = pd.concat(dfs, ignore_index=True)
                return df
            #
            return wrapper
        #
        return segment_op_

    #------------------------------------------------------------------------------------
    #季度数据
    @segment_op(limit=5000, _max=100000)
    def fina_indicator(self, *args, **kwargs):
        fields = '''ts_code,
        ann_date,
        end_date,
        eps,
        dt_eps,
        total_revenue_ps,
        revenue_ps,
        capital_rese_ps,
        surplus_rese_ps,
        undist_profit_ps,
        extra_item,
        profit_dedt,
        gross_margin,
        current_ratio,
        quick_ratio,
        cash_ratio,
        invturn_days,
        arturn_days,
        inv_turn,
        ar_turn,
        ca_turn,
        fa_turn,
        assets_turn,
        op_income,
        valuechange_income,
        interst_income,
        daa,
        ebit,
        ebitda,
        fcff,
        fcfe,
        current_exint,
        noncurrent_exint,
        interestdebt,
        netdebt,
        tangible_asset,
        working_capital,
        networking_capital,
        invest_capital,
        retained_earnings,
        diluted2_eps,
        bps,
        ocfps,
        retainedps,
        cfps,
        ebit_ps,
        fcff_ps,
        fcfe_ps,
        netprofit_margin,
        grossprofit_margin,
        cogs_of_sales,
        expense_of_sales,
        profit_to_gr,
        saleexp_to_gr,
        adminexp_of_gr,
        finaexp_of_gr,
        impai_ttm,
        gc_of_gr,
        op_of_gr,
        ebit_of_gr,
        roe,
        roe_waa,
        roe_dt,
        roa,
        npta,
        roic,
        roe_yearly,
        roa2_yearly,
        roe_avg,
        opincome_of_ebt,
        investincome_of_ebt,
        n_op_profit_of_ebt,
        tax_to_ebt,
        dtprofit_to_profit,
        salescash_to_or,
        ocf_to_or,
        ocf_to_opincome,
        capitalized_to_da,
        debt_to_assets,
        assets_to_eqt,
        dp_assets_to_eqt,
        ca_to_assets,
        nca_to_assets,
        tbassets_to_totalassets,
        int_to_talcap,
        eqt_to_talcapital,
        currentdebt_to_debt,
        longdeb_to_debt,
        ocf_to_shortdebt,
        debt_to_eqt,
        eqt_to_debt,
        eqt_to_interestdebt,
        tangibleasset_to_debt,
        tangasset_to_intdebt,
        tangibleasset_to_netdebt,
        ocf_to_debt,
        ocf_to_interestdebt,
        ocf_to_netdebt,
        ebit_to_interest,
        longdebt_to_workingcapital,
        ebitda_to_debt,
        turn_days,
        roa_yearly,
        roa_dp,
        fixed_assets,
        profit_prefin_exp,
        non_op_profit,
        op_to_ebt,
        nop_to_ebt,
        ocf_to_profit,
        cash_to_liqdebt,
        cash_to_liqdebt_withinterest,
        op_to_liqdebt,
        op_to_debt,
        roic_yearly,
        total_fa_trun,
        profit_to_op,
        q_opincome,
        q_investincome,
        q_dtprofit,
        q_eps,
        q_netprofit_margin,
        q_gsprofit_margin,
        q_exp_to_sales,
        q_profit_to_gr,
        q_saleexp_to_gr,
        q_adminexp_to_gr,
        q_finaexp_to_gr,
        q_impair_to_gr_ttm,
        q_gc_to_gr,
        q_op_to_gr,
        q_roe,
        q_dt_roe,
        q_npta,
        q_opincome_to_ebt,
        q_investincome_to_ebt,
        q_dtprofit_to_profit,
        q_salescash_to_or,
        q_ocf_to_sales,
        q_ocf_to_or,
        basic_eps_yoy,
        dt_eps_yoy,
        cfps_yoy,
        op_yoy,
        ebt_yoy,
        netprofit_yoy,
        dt_netprofit_yoy,
        ocf_yoy,
        roe_yoy,
        bps_yoy,
        assets_yoy,
        eqt_yoy,
        tr_yoy,
        or_yoy,
        q_gr_yoy,
        q_gr_qoq,
        q_sales_yoy,
        q_sales_qoq,
        q_op_yoy,
        q_op_qoq,
        q_profit_yoy,
        q_profit_qoq,
        q_netprofit_yoy,
        q_netprofit_qoq,
        equity_yoy,
        rd_exp,
        update_flag'''
        kwargs['fields'] = fields
        return self.pro.fina_indicator_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def income(self, *args, **kwargs):
        return self.pro.income_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def balancesheet(self, *args, **kwargs):
        return self.pro.balancesheet_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def cashflow(self, *args, **kwargs):
        return self.pro.cashflow_vip(*args, **kwargs)

    #------------------------------------------------------------------------------------
    #指数日行情
    def index_daily(self):
        index_list = ['000001.SH', '000300.SH', '000905.SH']
        tmp_dir = os.path.join(self.root, "__temp_index_daily__")
        for i in index_list:
            df = self.pro.index_daily(ts_code=i)
            path = os.path.join(tmp_dir, i+".csv")
            df.to_csv(path, encoding='gbk')
            print(i+".csv write ok !!!!!")

    #------------------------------------------------------------------------------------
    '''
    通过上面的函数,会从tushare把原始数据下载并保存到本地raw_data目录中
    raw_data/src目录: 股票基础列表,成交日列表
    raw_data/__temp_adj_factor__目录: 复权因子表(日频数据)
    raw_data/__temp_daily__目录: 每日行情表(日频数据)
    raw_data/__temp_daily_basic__目录: 每日指标表(日频数据)
    raw_data/__temp_limit_list__目录: 每日涨跌停表(日频数据)
    raw_data/__temp_moneyflow__目录: 每日个股资金流向表(日频数据)
    raw_data/__temp_suspend_d__目录: 每日停复牌表(日频数据)
    raw_data/__temp_index_daily__目录: 每日指数行情(日频数据)
    raw_data/__temp_balancesheet__目录: 资产负债表(季频数据)
    raw_data/__temp_cashflow__目录: 现金流量表(季频数据)
    raw_data/__temp_fina_indicator__目录: 财务指标表(季频数据)
    raw_data/__temp_income__目录: 利润表(季频数据)
    
    下面开始的函数主要就是通过上面这些原始数据生成一些月频基础指标,主要有三种形式:
    1. 通过 <日频数据> 生成 <月频指标>
    2. 通过 <季频数据> 生成 <月频指标>
    3. 通过 <日频数据>和<季频数据> 混合生成 <月频指标>
    '''

    def create_listday_matrix(self):
        ''' 股票上市存续周期日矩阵
        '''
        all_stocks_info = self.meta
        trade_days = self.tradedays

        def if_listed(series):
            nonlocal all_stocks_info
            code = series.name
            ipo_date = all_stocks_info.at[code, 'ipo_date']
            delist_date = all_stocks_info.at[code, 'delist_date']
            daterange = series.index
            if delist_date is pd.NaT:
                res = np.where(daterange >= ipo_date, 1, 0)
            else:
                res = np.where(daterange < ipo_date, 0, np.where(daterange <= delist_date, 1, 0))
            return pd.Series(res, index=series.index)

        listday_dat = pd.DataFrame(index=all_stocks_info.index, columns=trade_days)
        listday_dat = listday_dat.apply(if_listed, axis=1)
        self.close_file(listday_dat, 'listday_matrix')

    def create_month_tdays_begin_end(self, latest_month_end_tradeday=None):
        ''' 每月第一个和最后一个交易日映射
        '''
        tdays = self.tradedays
        months_start = tdays[0:1] + list(after_d for before_d, after_d in zip(tdays[:-1], tdays[1:]) if before_d.month != after_d.month)
        months_end = list(before_d for before_d, after_d in zip(tdays[:-1], tdays[1:]) if before_d.month != after_d.month) + tdays[-1:]
        if latest_month_end_tradeday is None:
            latest_month_end_tradeday = self.month_map.index[-1]
        if months_end[-1] > latest_month_end_tradeday:
            months_start, months_end = months_start[:-1], months_end[:-1]
        trade_days_be_month = pd.DataFrame(months_end, index=months_start, columns=['month_end'])
        trade_days_be_month.index.name = 'month_start'
        self.close_file(trade_days_be_month, 'trade_days_begin_end_of_month')

    def create_trade_status(self):
        ''' 股票停复牌状态
        '''
        tmp_dir = os.path.join(self.root, "__temp_suspend_d__")
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays)
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        df.loc[:, :] = 1 #默认都是正常状态
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col=[1], engine='python', encoding='gbk')
            df.loc[dat.index, tday] = 0 #停牌的设置为0
            print(tday)
        self.close_file(df, "trade_status")

    def create_maxupordown(self):
        ''' 股票涨跌停状态
        '''
        tmp_dir = os.path.join(self.root, "__temp_limit_list__")
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays)
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        df.loc[:, :] = 0 #默认都没有涨跌停
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col='ts_code', engine='python', encoding='gbk')
            #==================================================
            #有些股票已经名字和证劵代码,需要修改
            index = dat.index.to_series()
            index = index.replace("000022.SZ", "001872.SZ")
            index = index.replace("601313.SH", "601360.SH")
            index = index.replace("000043.SZ", "001914.SZ")
            #==================================================
            df.loc[index, tday] = 1 #涨跌停的设置为1
            print(tday)
        self.close_file(df, "maxupordown")

    def create_turn_d(self):
        ''' 日换手率
        '''
        self.create_indicator("__temp_daily_basic__", "turnover_rate", "turn")
        turn = self.turn / 100
        turn = self.preprocess(turn, suspend_days_process=True)
        self.close_file(turn, "turn")

    def create_mkt_cap_float_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["circ_mv"]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "mkt_cap_float_m")

    def create_pe_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["pe_ttm"]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "pe_ttm_m")

    def create_val_pe_deducted_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["pe"] #临时先用pe替代
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "val_pe_deducted_ttm_m")

    def create_pb_lf_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["pb"]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "pb_lf_m")

    def create_ps_ttm_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["ps_ttm"]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "ps_ttm_m")

    def create_pcf_ncf_ttm_m(self):
        s = pd.to_datetime('20090101') #统计周期开始
        e = pd.to_datetime('20191231') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta
        #-------------------------------------------------------
        #总市值指标(月频)
        df_total_mv = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays) #总市值指标(月频)
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__") #每日指标表
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df_total_mv[caldate] = dat["total_mv"]
            print(caldate)
        #df_total_mv = df_total_mv.dropna(how='all') #删掉全为空的一行
        print(df_total_mv) #总市值指标ok
        #-------------------------------------------------------
        #现金增加额指标(季频)
        tmp_dir = os.path.join(self.root, "__temp_cashflow__") #现金流量表
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        df_cfps = pd.DataFrame(index=all_stocks_info.index, columns=qdays) #现金增加额指标(季频)
        df_ann_date = pd.DataFrame(index=all_stocks_info.index, columns=qdays) #财报发布日期(季频)
        for qday in qdays:
            name = qday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date'])
            diff = dat.index.difference(df_cfps.index) #删除没在股票基础列表中多余的股票行
            dat = dat.loc[~dat.index.isin(diff)] #方法1
            #dat = dat.drop(labels=diff) #方法2
            #
            #x = dat.index.to_series()
            #print(x)
            #x = x.groupby(['ts_code'])
            #print(x)
            #print(x.count())
            #print(x.count()>1)
            #print(dat[x.count()>1])
            #
            #x = dat.index
            #print(x.duplicated())
            #print(dat[x.duplicated()])
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            df_cfps[qday] = dat["n_incr_cash_cash_equ"] #现金及现金等价物净增加额
            df_ann_date[qday] = dat["ann_date"] #财报发布日期
            print(qday)
        print(df_cfps) #现金增加额指标ok
        #df_cfps = df_cfps.dropna(how='all') #删掉全为空的一行
        #-------------------------------------------------------
        #现金增加额指标可能有空值,利用线性插值补全(这步可以不做)
        df_cfps_t = df_cfps.T #把时间变成索引,股票变成列名
        def _w(ser):
            if pd.isnull(ser[3]): #一年内如果第四季度(年报)指标值为空,那么整年四个季度都设置为空
                ser.iloc[:] = np.nan
            elif any(pd.isnull(ser)): #1~3季度如果存在空值,就利用线性插值补全
                if pd.isnull(ser[0]): #第一季度必须保证有值,才能进行插值
                    ser[0] = ser[3]/4 #第一季度如果为空,就用全年的均值进行填充
                ser = ser.interpolate()
            df_cfps_t.loc[ser.index, ser.name] = ser #回填
        df_cfps_t.resample('A').apply(_w) #按年分组处理
        df_cfps = df_cfps_t.T #变回来:股票为索引,日期为列名
        #-------------------------------------------------------
        #计算结果指标(月频)
        df_result = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        '''
        算法:
        (1)最新报告期是年报，则TTM=年报；
        (2)最新报告期不是年报，则TTM=本期+(上年年报-上年同期)，如果本期、上年年报、上年同期存在空值，则不计算，返回空值；
        (3)最新报告期通过财报发布时间进行判断,防止前视偏差。
        '''
        #按时间和股票逐个开始计算
        for calday in df_result.columns: #每月最后一天
            for stock in df_result.index:
                tmap = df_ann_date.loc[stock] #tmap索引为报告期(每季度最后一天),值为相应财报发布时间
                tmap = tmap[tmap<calday] #在那个历史节点,只能使用已经发布的财报,防止使用未来数据
                try:
                    d = tmap.index[-1] #已经发布的财报里面最近一期的时间(某季度最后一天)
                    if d.quarter == 4: #最近一期财报是年报(第4季度)
                        ttm_value = df_cfps.loc[stock, d]
                    else: #最近一期财报是1季度,2季度,或者3季度的情形
                        last_q_4 = tmap.index[-1-d.quarter] #相对于那一个历史节点的上一年年报的时间
                        last_q_same = tmap.index[-1-4] #相对于那一个历史节点的上一年同期的时间
                        ttm_value = df_cfps.loc[stock, d] + (df_cfps.loc[stock, last_q_4] - df_cfps.loc[stock, last_q_same]) #TTM=本期+(上年年报-上年同期)
                    #总市值/现金及现金等价物净增加额(TTM)
                    df_result.loc[stock, calday] = df_total_mv.loc[stock, calday]/ttm_value
                except:
                    pass
        df_result = df_result.dropna(how='all') #删掉全为空的一行
        self.close_file(df_result, "pcf_ncf_ttm_m")

    def create_pcf_ocf_ttm_m(self):
        ''' 本函数与上面的create_pcf_ncf_ttm_m类似,逻辑更优化
        '''
        s = pd.to_datetime('20090101') #统计周期开始
        e = pd.to_datetime('20191231') #统计周期结束
        new_tdays = self._get_trade_days(s, e, "M") #每月最后一个交易日
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays] #每月最后一天(每月最后一个日历日)
        all_stocks_info = self.meta
        #-------------------------------------------------------
        #总市值指标(月频)
        df_total_mv = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays) #总市值指标(月频)
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__") #每日指标表
        for tday in new_tdays: #每月最后一个交易日
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday] #每月最后一个日历日
            df_total_mv[caldate] = dat["total_mv"]
            print(caldate)
        df_total_mv = df_total_mv.dropna(how='all') #删掉全为空的一行
        #-------------------------------------------------------
        tmp_dir = os.path.join(self.root, "__temp_cashflow__") #现金流量表
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        panel = {}
        for d in qdays:
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.loc[~dat.index.isin(diff)]
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        panel = pd.Panel(panel)
        panel = panel.to_frame()
        panel = panel.stack().unstack(level=(0,1))
        #-------------------------------------------------------
        #开始计算结果指标(月频)
        df_result = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        '''
        算法:
        (1)最新报告期是年报，则TTM=年报；
        (2)最新报告期不是年报，则TTM=本期+(上年年报-上年同期)，如果本期、上年年报、上年同期存在空值，则不计算，返回空值；
        (3)最新报告期通过财报发布时间进行判断,防止前视偏差。
        '''
        #按时间和股票逐个开始计算
        for calday in df_result.columns: #每月最后一天
            for stock in df_result.index: #每只股票
                try:
                    datdf = panel[stock]
                    datdf = datdf.loc[datdf['ann_date']<calday] #在那个历史节点,只能使用已经发布的财报,防止使用未来数据
                    d = datdf.iloc[-1].name #已经发布的财报里面最近一期的时间(某季度最后一天)
                    if d.quarter == 4: #最近一期财报是年报(第4季度)
                        ttm_value = datdf.iloc[-1].at['n_cashflow_act']
                    else: #最近一期财报是1季度,2季度,或者3季度的情形
                        last_q_4 = datdf.iloc[-1-d.quarter] #相对于那一个历史节点的上一年年报
                        last_q_same = datdf.iloc[-1-4] #相对于那一个历史节点的上一年同期
                        #TTM=本期+(上年年报-上年同期)
                        ttm_value = datdf.iloc[-1].at['n_cashflow_act'] + (last_q_4.at['n_cashflow_act'] - last_q_same.at['n_cashflow_act'])
                    #总市值/经营活动产生的现金流量净额(TTM)
                    df_result.at[stock, calday] = df_total_mv.at[stock, calday]/ttm_value
                except:
                    pass
            print(calday)
        df_result = df_result.dropna(how='all') #删掉全为空的一行
        self.close_file(df_result, "pcf_ocf_ttm_m")

    def create_dividendyield2_m(self):
        ''' 通过日频数据创建月频指标(可统一为单个函数)
        '''
        tmp_dir = os.path.join(self.root, "__temp_daily_basic__")
        s = pd.to_datetime('20090101')
        e = pd.to_datetime('20191231')
        new_tdays = self._get_trade_days(s, e, "M")
        new_caldays = [self._get_month_end(tdate) for tdate in new_tdays]
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=new_caldays)
        for tday in new_tdays:
            name = tday.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk')
            caldate = self.month_map[tday]
            df[caldate] = dat["dv_ttm"]
            print(caldate)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "dividendyield2_m")

    def create_profit_ttm_G_m(self):
        ''' 通过季频数据创建月频指标,可以直接用create_indicator_m_by_q代替
        '''
        s = pd.to_datetime('20090101') #统计周期开始
        e = pd.to_datetime('20191231') #统计周期结束
        qdays = pd.date_range(start=s, end=e, freq="Q") #每个季度最后一天
        mdays = pd.date_range(start=s, end=e, freq="M") #每个月最后一天
        all_stocks_info = self.meta
        tmp_dir = os.path.join(self.root, "__temp_fina_indicator__") #财务指标表
        panel = {}
        for d in qdays: #每季度最后一天
            name = d.strftime("%Y%m%d")
            dat = pd.read_csv(os.path.join(tmp_dir, name+".csv"), index_col=[1], engine='python', encoding='gbk', parse_dates=['ann_date','end_date'])
            diff = dat.index.difference(all_stocks_info.index) #删除没在股票基础列表中多余的股票行
            dat = dat.drop(labels=diff)
            dat = dat[~dat.index.duplicated(keep='last')] #财务数据中同一只股票可能会有重复的记录,删除多余重复的
            del dat['Unnamed: 0']
            panel[d] = dat
            print(d)
        panel = pd.Panel(panel)
        panel = panel.to_frame()
        '''
                                             2009-03-31           2009-06-30           2009-09-30           2009-12-31           2010-03-31           2010-06-30           2010-09-30           2010-12-31           2011-03-31           2011-06-30           2011-09-30           2011-12-31           2012-03-31           2012-06-30           2012-09-30           2012-12-31           2013-03-31           2013-06-30           2013-09-30           2013-12-31           2014-03-31           2014-06-30           2014-09-30           2014-12-31           2015-03-31           2015-06-30           2015-09-30           2015-12-31           2016-03-31           2016-06-30           2016-09-30           2016-12-31           2017-03-31           2017-06-30           2017-09-30           2017-12-31           2018-03-31           2018-06-30           2018-09-30           2018-12-31           2019-03-31           2019-06-30           2019-09-30           2019-12-31
major     minor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
000001.SZ ann_date          2009-04-24 00:00:00  2009-08-21 00:00:00  2009-10-29 00:00:00  2010-03-12 00:00:00  2010-04-29 00:00:00  2010-08-25 00:00:00  2010-10-28 00:00:00  2011-02-25 00:00:00  2011-04-27 00:00:00  2011-08-18 00:00:00  2011-10-26 00:00:00  2012-03-09 00:00:00  2012-04-26 00:00:00  2012-08-16 00:00:00  2012-10-26 00:00:00  2013-03-08 00:00:00  2013-04-24 00:00:00  2013-08-23 00:00:00  2013-10-23 00:00:00  2014-03-07 00:00:00  2014-04-24 00:00:00  2014-08-14 00:00:00  2014-10-24 00:00:00  2015-03-13 00:00:00  2015-04-24 00:00:00  2015-08-14 00:00:00  2015-10-23 00:00:00  2016-03-10 00:00:00  2016-04-21 00:00:00  2016-08-12 00:00:00  2016-10-21 00:00:00  2017-03-17 00:00:00  2017-04-22 00:00:00  2017-08-11 00:00:00  2017-10-21 00:00:00  2018-03-15 00:00:00  2018-04-20 00:00:00  2018-08-16 00:00:00  2018-10-24 00:00:00  2019-03-07 00:00:00  2019-04-24 00:00:00  2019-08-08 00:00:00  2019-10-22 00:00:00  2020-02-14 00:00:00
          end_date          2009-03-31 00:00:00  2009-06-30 00:00:00  2009-09-30 00:00:00  2009-12-31 00:00:00  2010-03-31 00:00:00  2010-06-30 00:00:00  2010-09-30 00:00:00  2010-12-31 00:00:00  2011-03-31 00:00:00  2011-06-30 00:00:00  2011-09-30 00:00:00  2011-12-31 00:00:00  2012-03-31 00:00:00  2012-06-30 00:00:00  2012-09-30 00:00:00  2012-12-31 00:00:00  2013-03-31 00:00:00  2013-06-30 00:00:00  2013-09-30 00:00:00  2013-12-31 00:00:00  2014-03-31 00:00:00  2014-06-30 00:00:00  2014-09-30 00:00:00  2014-12-31 00:00:00  2015-03-31 00:00:00  2015-06-30 00:00:00  2015-09-30 00:00:00  2015-12-31 00:00:00  2016-03-31 00:00:00  2016-06-30 00:00:00  2016-09-30 00:00:00  2016-12-31 00:00:00  2017-03-31 00:00:00  2017-06-30 00:00:00  2017-09-30 00:00:00  2017-12-31 00:00:00  2018-03-31 00:00:00  2018-06-30 00:00:00  2018-09-30 00:00:00  2018-12-31 00:00:00  2019-03-31 00:00:00  2019-06-30 00:00:00  2019-09-30 00:00:00  2019-12-31 00:00:00
          eps                              0.36                 0.74                 1.17                 1.62                 0.51                 0.98                 1.46                 1.91                 0.69                 1.36                 2.01                 2.47                 0.67                 1.32                    2                 2.62                  0.7                 0.92                 1.43                 1.86                 0.53                 0.88                 1.37                 1.73                 0.41                 0.84                 1.27                 1.56                 0.43                 0.72                 1.09                 1.32                 0.31                 0.68                 1.06                  1.3                 0.33                 0.73                 1.14                 1.39                 0.38                 0.85                 1.32                 1.54
          dt_eps                           0.36                 0.74                 1.17                 1.62                 0.51                 0.98                 1.46                 1.91                 0.69                 1.36                 2.01                 2.47                 0.67                 1.32                    2                 2.62                  0.7                 0.92                 1.43                 1.86                 0.53                 0.88                 1.37                 1.73                 0.41                 0.84                 1.27                 1.56                 0.43                 0.72                 1.09                 1.32                 0.31                 0.68                 1.06                  1.3                 0.33                 0.73                 1.14                 1.39                 0.36                 0.78                 1.32                 1.45
          total_revenue_ps                1.211               2.4122               3.5789               4.8671               1.3152               2.4379               3.7761               5.1714               1.6686               3.4837               4.0406               5.7859                1.898               3.8306               5.7641               7.7583               2.1085               2.8579               4.5559               5.4815                1.691               3.0401               4.7835               6.4251               1.8093               3.2549               4.9725               6.7205               1.9241               3.1898               4.7739               6.2734                1.614               3.1493               4.6496               6.1611               1.6323               3.3338               5.0474               6.7977               1.8914               3.9504               5.3055                7.109
          revenue_ps                      1.211               2.4122               3.5789               4.8671               1.3152               2.4379               3.7761               5.1714               1.6686               3.4837               4.0406               5.7859                1.898               3.8306               5.7641               7.7583               2.1085               2.8579               4.5559               5.4815                1.691               3.0401               4.7835               6.4251               1.8093               3.2549               4.9725               6.7205               1.9241               3.1898               4.7739               6.2734                1.614               3.1493               4.6496               6.1611               1.6323               3.3338               5.0474               6.7977               1.8914               3.9504               5.3055                7.109
          capital_rese_ps                2.4242                2.336               2.2635               2.2596               2.2796               3.8898               3.8961               3.8442               3.8542               3.8177               7.9101               8.1075               8.1223               8.0588               7.8517               7.8338               7.9069               4.9067               4.8098               5.4337                5.451               4.3886               4.5751               4.5751               4.5751               4.1461               4.1461               4.1461               4.1461               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               3.2886               4.1645               4.1645
          surplus_rese_ps                0.2515               0.2515               0.2515               0.4135               0.4135               0.3684               0.3684               0.5487               0.5487               0.5487               0.3733               0.5525               0.5525               0.5525               0.5525               0.5525               0.5524               0.3452               0.3452               0.4573               0.4573               0.3811               0.3811               0.5544               0.5544               0.4427               0.4427               0.5955               0.5955               0.4963               0.4963               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.6279               0.5555               0.5555
          undist_profit_ps               0.6679               1.0509               1.4779               1.4126               1.9208               2.1291               2.6172               2.5081               3.1974               3.8659               3.1558               3.0965               3.7658               3.6659               4.2444               4.5042               5.2052               3.2527               3.7608                3.147               3.6779               3.3709               3.8629               3.8211               4.3138               3.7216               4.1517               3.6993               4.1246               3.6713               4.0456               3.7358               4.0468                4.258               4.6423               4.6395               4.6852                4.944               5.3566               5.5351               5.9178               6.2362               5.9411                5.842
          extra_item                  1.897e+06            5.014e+06           1.5264e+07           9.1158e+07            1.519e+06           5.6565e+07           7.8095e+07          1.40079e+08            7.155e+06           1.3668e+07           8.3286e+07           9.9359e+07            8.733e+06           3.2224e+07           5.0513e+07           1.7481e+07                1e+06                9e+06              5.9e+07              6.5e+07             -1.1e+07             -1.3e+07             -1.7e+07             -3.9e+07               -8e+06               -6e+06             -1.4e+07             -3.7e+07              1.1e+07               -2e+06              1.2e+07               -7e+06                2e+06              4.2e+07              2.3e+07              2.7e+07                4e+07              4.6e+07             1.09e+08             1.18e+08              2.4e+07              8.7e+07              9.4e+07             1.09e+08
        '''
        #print(panel.iloc[:10,:100])
        panel = panel.stack().unstack(level=(0,1))
        '''
        major                 000001.SZ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      000002.SZ                                                                               
minor                  ann_date             end_date   eps dt_eps total_revenue_ps revenue_ps capital_rese_ps surplus_rese_ps undist_profit_ps   extra_item  profit_dedt assets_turn    op_income valuechange_income retained_earnings diluted2_eps      bps  ocfps retainedps    cfps netprofit_margin profit_to_gr adminexp_of_gr op_of_gr      roe   roe_dt    npta roe_yearly opincome_of_ebt investincome_of_ebt n_op_profit_of_ebt tax_to_ebt dtprofit_to_profit ocf_to_or ocf_to_opincome debt_to_assets assets_to_eqt dp_assets_to_eqt debt_to_eqt eqt_to_debt ocf_to_debt roa_yearly  roa_dp fixed_assets non_op_profit op_to_ebt nop_to_ebt ocf_to_profit op_to_debt total_fa_trun profit_to_op   q_opincome q_investincome   q_dtprofit   q_eps q_netprofit_margin q_profit_to_gr q_adminexp_to_gr q_op_to_gr   q_roe q_dt_roe  q_npta q_opincome_to_ebt q_investincome_to_ebt q_dtprofit_to_profit q_ocf_to_sales q_ocf_to_or basic_eps_yoy dt_eps_yoy cfps_yoy   op_yoy  ebt_yoy netprofit_yoy dt_netprofit_yoy  ocf_yoy  roe_yoy bps_yoy assets_yoy  eqt_yoy   tr_yoy   or_yoy q_gr_yoy q_gr_qoq q_sales_yoy q_sales_qoq q_op_yoy q_op_qoq q_profit_yoy q_profit_qoq q_netprofit_yoy q_netprofit_qoq equity_yoy update_flag             ann_date             end_date    eps dt_eps total_revenue_ps revenue_ps capital_rese_ps
2009-03-31  2009-04-24 00:00:00  2009-03-31 00:00:00  0.36   0.36            1.211      1.211          2.4242          0.2515           0.6679    1.897e+06  1.12018e+09      0.0075  1.27687e+09         2.3676e+08       2.85515e+09       0.3613   5.4975   0.87     0.9194 -0.0199          29.8377      29.8377        38.9747  40.2498   6.7044   6.6931  0.2252    26.8176         84.1643             15.6059             0.2297    26.0389            99.8309    0.7197          2.1195        96.7287       30.5689          29.7649     29.5691      0.0338      0.0054     0.9008  0.2252  2.05976e+09     3.486e+06   99.7702     0.2298       178.801      0.003        1.8139      40.3425  1.27687e+09     2.3676e+08  1.12018e+09  0.3613            29.8377        29.8377          38.9747    40.2498  6.7044   6.6931  0.2252           84.1643               15.6059              99.8309        71.9669     211.954        5.8824     9.0909    -47.9   9.9744  10.0076       11.7404          11.8029 -29.2649  -8.0964    4.09      9.999   4.0933    5.835    5.835    5.835  -0.2945       5.835     -0.2945   9.9744  142.508      11.7404      141.513         11.7404         141.513    21.5844           1  2009-04-27 00:00:00  2009-03-31 00:00:00   0.07   0.07           0.7425     0.7425           0.717
2009-06-30  2009-08-21 00:00:00  2009-06-30 00:00:00  0.74   0.74           2.4122     2.4122           2.336          0.2515           1.0509    5.014e+06  2.30637e+09      0.0148  2.33276e+09        5.88767e+08       4.04447e+09       0.7443   5.7922   1.24     1.3024   2.145          30.8562      30.8562        39.2871  39.0013  13.4429  13.4138  0.4551    26.8858         79.6306              20.098             0.2714     21.099            99.7831    0.5133          1.6482        96.6765       30.0888          29.5353     29.0892      0.0344      0.0073     0.9102  0.4551  2.07566e+09      7.95e+06   99.7286     0.2714       131.605     0.0056        3.5994      39.1074  1.05589e+09    3.52007e+08  1.18619e+09   0.383            31.8829        31.8829           39.602    37.7427  6.7845   6.7667  0.2237           74.7606               24.9233              99.7379        30.5202     107.822        4.2254     5.7143   -68.12   2.8347   3.6739        7.8156             7.05 -58.5993   1.5574    9.67    14.0767   9.6739   5.2788   5.2788   4.7239  -0.8073      4.7239     -0.8073  -3.8747  -6.9858       4.3574       5.9919          4.3574          5.9919     6.1623           1  2009-08-04 00:00:00  2009-06-30 00:00:00   0.23   0.23           1.9835     1.9835          0.7738
2009-09-30  2009-10-29 00:00:00  2009-09-30 00:00:00  1.17   1.17           3.5789     3.5789          2.2635          0.2515           1.4779   1.5264e+07  3.62216e+09      0.0216  3.84681e+09        6.79641e+08        5.3705e+09       1.1713   6.1468  -0.08     1.7294  2.1808          32.7284      32.7284         39.985  40.7276  20.4987  20.4127  0.7072    27.3316         84.7195             14.9679             0.3126     19.892            99.5804   -0.0232          -0.067        96.5561       29.0368          28.9864     28.0367      0.0357     -0.0005     0.9429  0.7072  2.24311e+09    1.4193e+07   99.6874     0.3126       -5.6961     0.0085        5.1338      40.8553  1.51405e+09     9.0873e+07  1.31578e+09   0.427            36.5993        36.5993           41.428    44.2968  7.1531   7.0978  0.2421           93.9723                5.6402               99.227       -113.237    -270.976        7.3394     8.3333  -100.48   3.7176   5.2373        9.6601           7.8789 -100.652   5.5596   16.39     16.825  16.3873   3.4683   3.4683  -0.0841  -2.8723     -0.0841     -2.8723   5.3645  13.9943      13.0307      11.4957         13.0307         11.4957     3.8845           1  2009-10-26 00:00:00  2009-09-30 00:00:00  0.269  0.269            2.687      2.687          0.7728
2009-12-31  2010-03-12 00:00:00  2009-12-31 00:00:00  1.62   1.62           4.8671     4.8671          2.2596          0.4135           1.4126   9.1158e+07  4.93957e+09      0.0285  5.33855e+09        8.20577e+08       5.67083e+09         1.62   6.5915  10.37     1.8261  5.6607          33.2843      33.2843        41.7554    40.75  27.2887  26.7942  0.9472    27.2887         86.2373             13.2553             0.5074    18.7352             98.188      2.13          6.0304        96.5177       28.7167          28.8104     27.7163      0.0361      0.0567     0.9472  0.9472  2.23831e+09     3.141e+07   99.4926     0.5074       522.698     0.0109        6.9895      40.9578  1.49174e+09    1.40936e+08  1.31742e+09  0.4487            34.8285        34.8285          46.6737     40.812  7.0444   6.6607   0.244           90.4142                8.5421               94.553        811.187     2175.41           710        710    32.25  666.608  681.033        719.29           691.67  32.2521  556.437   24.81    23.8957  24.8087   4.1433   4.1433   6.0655   10.416      6.0655      10.416  145.852   1.7296      151.548       5.0737         151.548          5.0737    24.8087           1  2010-03-02 00:00:00  2009-12-31 00:00:00   0.48   0.48           4.4457     4.4457          0.7783
2010-03-31  2010-04-29 00:00:00  2010-03-31 00:00:00  0.51   0.51           1.3152     1.3152          2.2796          0.4135           1.9208    1.519e+06   1.5766e+09      0.0068  1.86759e+09        1.27562e+08       7.24894e+09       0.5082   7.1197   2.77     2.3343  0.4794          38.6391      38.6391        41.2532  48.8499   7.4126   7.4054  0.2613    29.6504         93.5187              6.3876             0.0936    20.9764            99.9037     2.108            4.61        96.4335       28.0387          28.3644     27.0386       0.037      0.0144     1.0452  0.2613  2.45558e+09      1.87e+06   99.9064     0.0936       431.528     0.0033        1.7402      48.8956  1.86759e+09    1.27562e+08   1.5766e+09  0.5082            38.6391        38.6391          41.2532    48.8499  7.4126   7.4054  0.2613           93.5187                6.3876              99.9037        210.801     461.003       41.6667    41.6667   218.12   31.812  31.6325       40.6426          40.7451  218.123   8.5973    8.01     5.4638   8.0129   8.6063   8.6063   8.6063   2.0939      8.6063      2.0939   31.812  22.2011      40.6426       13.264         40.6426          13.264    29.5083           1  2010-04-27 00:00:00  2010-03-31 00:00:00    0.1    0.1           0.6826     0.6826          0.8215
2010-06-30  2010-08-25 00:00:00  2010-06-30 00:00:00  0.98   0.98           2.4379     2.4379          3.8898          0.3684           2.1291   5.6565e+07  2.97655e+09       0.014  3.54126e+09        2.63571e+08       8.70395e+09       0.8703   8.7291  -0.98     2.4975   0.459          35.7001      35.7001        42.2603  44.7832  11.9201  11.6978  0.5004    23.8402         91.4641              6.8075             1.7284    21.6602            98.1351    -0.357         -0.8566        95.1279        20.525          23.8198     19.5252      0.0512     -0.0051     1.0008  0.5004  2.68418e+09    6.6919e+07   98.2716     1.7284       -79.726     0.0064         3.452      45.5708  1.67367e+09    1.36009e+08  1.39996e+09  0.4175            32.9793        32.9793          43.1926    41.0185  5.5396     5.33  0.2339           89.2753                7.2549              96.2168       -263.904    -695.662       32.4324    32.4324   -178.9  30.2341   32.165        31.225            29.06 -178.896 -22.4093   32.47     6.2243   48.616  13.4199  13.4199  18.2726   8.0212     18.2726      8.0212  28.5378  -9.2962      22.3398      -7.8015         22.3398         -7.8015    69.1246           1  2010-08-10 00:00:00  2010-06-30 00:00:00   0.26   0.26           1.5249     1.5249          0.8063
2010-09-30  2010-10-28 00:00:00  2010-09-30 00:00:00  1.46   1.46           3.7761     3.7761          3.8961          0.3684           2.6172   7.8095e+07  4.65584e+09      0.0208   5.4743e+09          4.452e+08       1.04048e+10       1.3584   9.2235   3.04     2.9856  2.7357          35.9732      35.9732        41.7209  44.9822  17.9951  17.6982  0.7497    23.9935         91.0283              7.4029             1.5688    21.2826            98.3503     0.804          1.9326        95.2384       21.0013          24.0028     20.0012        0.05      0.0165     0.9996  0.7497  2.93323e+09    9.4345e+07   98.4312     1.5688       178.728     0.0092        5.0893      45.6992  1.93304e+09    1.81629e+08  1.67929e+09   0.488            36.4707        36.4707          40.7383    45.3449   5.437   5.3681  0.2618           90.2407                 8.479              98.7341        291.908     704.238       24.7863    24.7863     3900  30.7758  32.4448       30.1456           28.538  4203.35 -22.7145   39.91    14.8437  57.0335  18.4065  18.4065  28.7164   5.7044     28.7164      5.7044  31.7618  16.8536       28.264      16.8948          28.264         16.8948    68.3959           1  2010-10-25 00:00:00  2010-09-30 00:00:00  0.298  0.298           2.0355     2.0355          0.8052
2010-12-31  2011-02-25 00:00:00  2010-12-31 00:00:00  1.91   1.91           5.1714     5.1714          3.8442          0.5487           2.5081  1.40079e+08  6.14374e+09      0.0274  7.38911e+09        4.60446e+08       1.06531e+10       1.8031   9.6163   6.24     3.0568  3.8578          34.8669      34.8669        40.8384  43.5547  23.2809   22.762  0.9554    23.2809         92.3867               5.757             1.8563    21.4329            97.7708    1.2066           2.943        95.3941       21.7113          24.3676     20.7114      0.0483      0.0313     0.9554  0.9554  3.00986e+09   1.48466e+08   98.1437     1.8563       277.039     0.0113         6.868      44.3785  1.91481e+09     1.5246e+07  1.48789e+09  0.4447            31.8731        31.8731          38.4501    39.6915  4.7211   4.5323   0.221           96.5039                0.7684              96.0007        229.639     583.166       17.9012    17.9012   -39.83  27.4459  29.1975       24.9087            24.38 -32.4515  -23.706   45.98     23.783  63.7202  19.2388  19.2388  21.5512   4.2695     21.5512      4.2695  18.2141  -8.7303       11.237      -8.8748          11.237         -8.8748    63.7202           1  2011-03-08 00:00:00  2010-12-31 00:00:00   0.66   0.66           4.6124     4.6124          0.7994
2011-03-31  2011-04-27 00:00:00  2011-03-31 00:00:00  0.69   0.69           1.6686     1.6686          3.8542          0.5487           3.1974    7.155e+06  2.39504e+09      0.0076  2.86253e+09        1.97425e+08       1.30553e+10       0.6893  10.3156   6.13     3.7461  3.7853          41.3101      41.3101        35.5607  52.6214   6.9165   6.8959   0.313     27.666         93.2695              6.4327             0.2979    21.7295            99.7021    3.6709          7.4572        95.5483       22.4633          22.1006     21.4635      0.0466      0.0277      1.252   0.313  2.94687e+09     9.142e+06   99.7021     0.2979        697.61      0.004        1.9524      52.7787  2.86253e+09    1.97425e+08  2.39504e+09  0.6893            41.3101        41.3101          35.5607    52.6214  6.9165   6.8959   0.313           93.2695                6.4327              99.7021        367.092     745.723       35.2941    35.2941    121.3  53.3697  53.6839       52.2192          51.9121  147.938  -6.3829    7.27      11.05   8.2914  42.3771  42.3771  42.3771  19.5861     42.3771     19.5861  53.3697  58.5425      52.2192       54.993         52.2192          54.993    62.5977           1  2011-04-20 00:00:00  2011-03-31 00:00:00   0.11   0.11           0.7249     0.7249          0.8044
2011-06-30  2011-08-18 00:00:00  2011-06-30 00:00:00  1.36   1.36           3.4837     3.4837          3.8177          0.5487           3.8659   1.3668e+07  4.71808e+09      0.0154  5.55363e+09        4.96266e+08       1.53849e+10       1.3577  10.9476   9.06     4.4146  3.7561          38.9746      38.9746        36.2991   49.832  13.2051   13.167  0.5991    26.4102         91.5557              8.1813              0.263    21.9936            99.7111    2.6007          5.6853        95.5223       22.3329          22.0423      21.333      0.0469      0.0388     1.1982  0.5991  2.89904e+09    1.5952e+07    99.737      0.263       521.892     0.0074        4.1093      49.9634   2.6911e+09    2.98841e+08  2.32303e+09  0.6684            36.8276        36.8276          36.9779    47.2676  6.2874   6.2698  0.2807           89.8006                9.9722              99.7204        161.683     380.044       38.7755    38.7755  1024.49  59.0057  56.6695       56.0026            58.51  1140.86  24.3899   13.84    17.1684  14.9254  42.8958  42.8958  43.3761   8.7792     43.3761      8.7792  65.2194  -2.2883      60.1062      -3.0244         60.1062         -3.0244    25.4142           1  2011-08-09 00:00:00  2011-06-30 00:00:00   0.27   0.27            1.818      1.818          0.7998
        '''
        #print(panel.iloc[:10,:100])
        #开始计算结果指标(月频)
        df = pd.DataFrame(index=all_stocks_info.index, columns=mdays)
        for d in df.columns: #每月最后一天
            #站在当前时间节点,每只股票所能看到的最近一期财务指标数据(不同股票财报发布时间不一定相同)
            for stock in df.index: #每只股票
                try:
                    datdf = panel[stock]
                    datdf = datdf.loc[datdf['ann_date']<d]
                    df.at[stock, d] = datdf.iloc[-1].at['q_profit_yoy']
                except:
                    pass
            print(d)
        df = df.dropna(how='all') #删掉全为空的一行
        self.close_file(df, "profit_ttm_G_m")


class WindFetcher(RawDataFetcher):

    def __init__(self):
        self.conn = pymysql.Connect(host='x', user='x', password='x', db='wind', charset='gbk')
        super().__init__(using_fetch=True)

    def daily(self, t):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHAREEODPRICES where TRADE_DT like '%s'" % t
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","TRADE_DT":"trade_date","S_DQ_CLOSE":"close","S_DQ_PCTCHANGE":"pct_chg","S_DQ_VOLUME":"vol","S_DQ_AMOUNT":"amount"})
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def daily_basic(self, t):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHAREEODDERIVATIVEINDICATOR where TRADE_DT like '%s'" % t
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","TRADE_DT":"trade_date"})
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def fina_indicator(self, period):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHAREFINANCIALINDICATOR where REPORT_PERIOD like '%s'" % period
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","ANN_DT":"ann_date","REPORT_PERIOD":"end_date"})
            del df['WIND_CODE']
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def fina_indicator_ttm(self, period):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHARETTMHIS where REPORT_PERIOD like '%s'" % period
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","ANN_DT":"ann_date","REPORT_PERIOD":"end_date"})
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def income(self, period):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHAREINCOME where REPORT_PERIOD like '%s'" % period
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","ANN_DT":"ann_date","REPORT_PERIOD":"end_date"})
            del df['WIND_CODE']
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def balancesheet(self, period):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHAREBALANCESHEET where REPORT_PERIOD like '%s'" % period
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","ANN_DT":"ann_date","REPORT_PERIOD":"end_date"})
            del df['WIND_CODE']
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def cashflow(self, period):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHARECASHFLOW where REPORT_PERIOD like '%s'" % period
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code","ANN_DT":"ann_date","REPORT_PERIOD":"end_date"})
            del df['WIND_CODE']
            del df['CRNCY_CODE']
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']
        return df

    def suspend_d(self, t):
        with self.conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM ASHARETRADINGSUSPENSION where S_DQ_SUSPENDDATE like '%s'" % t
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            df = df.rename(columns={"S_INFO_WINDCODE":"ts_code"})
            del df['OBJECT_ID']
            del df['OPDATE']
            del df['OPMODE']            
        return df

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    def create_trade_status(self):
        ''' 股票停复牌状态
        '''
        tmp_dir = os.path.join(self.root, "__temp_suspend_d__")
        tdays = [pd.to_datetime(f.split(".")[0]) for f in os.listdir(tmp_dir)]
        tdays = sorted(tdays)
        all_stocks_info = self.meta
        df = pd.DataFrame(index=all_stocks_info.index, columns=tdays)
        df.loc[:, :] = 1 #默认都是正常状态
        for f in os.listdir(tmp_dir):
            tday = pd.to_datetime(f.split(".")[0])
            dat = pd.read_csv(os.path.join(tmp_dir, f), index_col=['ts_code'], engine='python', encoding='gbk')
            df.loc[dat.index, tday] = 0 #停牌的设置为0
            print(tday)
        self.close_file(df, "trade_status")

    def create_profit_ttm_G_m(self):
        self.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "NET_PROFIT_TTM", "profit_ttm_m")
        profit_ttm_G_m = self.profit_ttm_m.T / self.profit_ttm_m.T.shift(12) - 1 #分母为0的情况导致产生inf值,不过没关系,最后生成因子截面时候统一处理
        profit_ttm_G_m = profit_ttm_G_m.T.dropna(how='all', axis=1)
        self.close_file(profit_ttm_G_m, "profit_ttm_G_m")
        print("'profit_ttm_G_m' updated.")

    def create_qfa_roe_G_m(self):
        self.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_ROE", "qfa_roe_m")
        qfa_roe_G_m = self.qfa_roe_m.T / self.qfa_roe_m.T.shift(12) - 1 #分母为0的情况导致产生inf值,不过没关系,最后生成因子截面时候统一处理
        qfa_roe_G_m = qfa_roe_G_m.T.dropna(how='all', axis=1)
        self.close_file(qfa_roe_G_m, "qfa_roe_G_m")
        print("'qfa_roe_G_m' updated.")
