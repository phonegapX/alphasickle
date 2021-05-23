# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
from datetime import datetime
from functools import reduce
from itertools import dropwhile
warnings.filterwarnings('ignore')

WORK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")

class FileAlreadyExistError(Exception):
    pass

class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

class Data:
    startday = "20090101"
    endday = "20191231"
    #endday = pd.tseries.offsets.datetime.now().strftime("%Y%m%d")
    freq = "M"

    root = WORK_PATH
    metafile = 'all_stocks.xlsx'
    mmapfile = 'month_map.xlsx'
    month_group_file = 'month_group.xlsx'
    tradedays_file = 'tradedays.xlsx'
    tdays_be_m_file = 'trade_days_begin_end_of_month.xlsx'

    value_indicators = [
            'pe_ttm', 'val_pe_deducted_ttm', 'pb_lf', 'ps_ttm',
            'pcf_ncf_ttm', 'pcf_ocf_ttm', 'dividendyield2', 'profit_ttm'
            ]
    value_target_indicators = [
            "EP", "EPcut", "BP", "SP",
            "NCFP", "OCFP", "DP", "G/PE"
            ]

    growth_indicators = [
            "qfa_yoysales", "qfa_yoyprofit", "qfa_yoyocf", "qfa_roe_G_m"
            ]
    growth_target_indicators = [
            "Sales_G_q", "Profit_G_q", "OCF_G_q", "ROE_G_q"
            ]

    finance_indicators = [
            "roe_ttm2_m", "qfa_roe_m",
            "roa2_ttm2_m", "qfa_roa_m",
            "grossprofitmargin_ttm2_m", "qfa_grossprofitmargin_m",
            "deductedprofit_ttm", "qfa_deductedprofit_m", "or_ttm", "qfa_oper_rev_m",
            "turnover_ttm_m", "qfa_netprofitmargin_m",
            "ocfps_ttm", "eps_ttm", "qfa_net_profit_is_m", "qfa_net_cash_flows_oper_act_m"
            ]
    finance_target_indicators = [
            "ROE_q", "ROE_ttm",
            "ROA_q", "ROA_ttm",
            "grossprofitmargin_q", "grossprofitmargin_ttm",
            "profitmargin_q", "profitmargin_ttm",
            "assetturnover_q", "assetturnover_ttm",
            "operationcashflowratio_q", "operationcashflowratio_ttm"
            ]

    leverage_indicators = [
            "assetstoequity_m", "longdebttoequity_m",
            "cashtocurrentdebt_m", "current_m"
            ]
    leverage_target_indicators = [
            "financial_leverage", "debtequityratio",
            "cashratio", "currentratio"
            ]

    cal_indicators = ["mkt_cap_float", "holder_avgpct", "holder_num"]
    cal_target_indicators = [
            "ln_capital",
            "HAlpha", "return_1m", "return_3m", "return_6m", "return_12m",
            "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
            "exp_wgt_return_1m",  "exp_wgt_return_3m",  "exp_wgt_return_6m", "exp_wgt_return_12m",
            "std_1m", "std_3m", "std_6m", "std_12m",
            "beta",
            "turn_1m", "turn_3m", "turn_6m", "turn_12m",
            "bias_turn_1m", "bias_turn_3m", "bias_turn_6m", "bias_turn_12m",
            "holder_avgpctchange",
            ]

    tech_indicators = [
            "MACD", "RSI", "PSY", "BIAS"
            ]
    tech_target_indicators = [
            "MACD", "DEA", "DIF", "RSI", "PSY", "BIAS"
            ]

    barra_quote_indicators = [
            "mkt_cap_float", "pct_chg", "amt"
            ]
    barra_quote_target_indicators = [
            "LNCAP_barra", "MIDCAP_barra",
            "BETA_barra", "HSIGMA_barra", "HALPHA_barra",
            "DASTD_barra", "CMRA_barra",
            "STOM_barra", "STOQ_barra", "STOA_barra",
            "RSTR_barra"
            ]

    barra_finance_indicators = [
            "mkt_cap_ard", "longdebttodebt", "other_equity_instruments_PRE",
            "tot_equity", "tot_liab", "tot_assets", "pb_lf",
            "pe_ttm", "pcf_ocf_ttm", "eps_ttm", "orps"
            ]
    barra_finance_target_indicators = [
            "MLEV_barra", "BLEV_barra", "DTOA_barra", "BTOP_barra",
            "ETOP_barra", "CETOP_barra", "EGRO_barra", "SGRO_barra"
            ]

    _tech_params = {
                    "BIAS": [20],
                    "MACD": [10, 30, 15],
                    "PSY": [20],
                    "RSI": [20],
                    }
    freqmap = {}

    def __init__(self):
        self.__update_frepmap()

    def __update_frepmap(self):
        self.freqmap.update({name.split(".")[0]: self.root for name in os.listdir(self.root)})

    def open_file(self, name):
        if name == 'meta':
            return pd.read_excel(os.path.join(self.root, 'src', self.metafile), index_col=[0], parse_dates=['ipo_date', "delist_date"], encoding='gbk')
        elif name == 'month_map':
            return pd.read_excel(os.path.join(self.root, 'src', self.mmapfile), index_col=[0], parse_dates=[0, 1], encoding='gbk')['calendar_date']
        elif name == 'trade_days_begin_end_of_month':
            return pd.read_excel(os.path.join(self.root, 'src', self.tdays_be_m_file), index_col=[1], parse_dates=[0, 1], encoding='gbk')
        elif name == 'month_group':
            return pd.read_excel(os.path.join(self.root, 'src', self.month_group_file), index_col=[0], parse_dates=True, encoding='gbk')
        elif name == 'tradedays':
            return pd.read_excel(os.path.join(self.root, 'src', self.tradedays_file), index_col=[0], parse_dates=True, encoding='gbk').index.tolist()
        path = self.freqmap.get(name, None)
        if path is None:
            raise Exception(f'{name} is unrecognisable or not in file dir, please check and retry.')
        try:
            dat = pd.read_csv(os.path.join(path, name+'.csv'), index_col=[0], engine='python', encoding='gbk')
            dat = pd.DataFrame(data=dat, index=dat.index.union(self.meta.index), columns=dat.columns)
        except TypeError:
            print(name, path)
            raise
        dat.columns = pd.to_datetime(dat.columns)
        #if name in ('stm_issuingdate', 'applied_rpt_date_M'):
        #    dat = dat.replace('0', np.nan)
        #    dat = dat.applymap(pd.to_datetime)
        return dat

    def close_file(self, df, name, **kwargs):
        if name == 'meta':
            df.to_excel(os.path.join(self.root, 'src', self.metafile), encoding='gbk', **kwargs)
        elif name == 'month_map':
            df.to_excel(os.path.join(self.root, 'src', self.mmapfile), encoding='gbk', **kwargs)
        elif name == 'trade_days_begin_end_of_month':
            df.to_excel(os.path.join(self.root, 'src', self.tdays_be_m_file), encoding='gbk', **kwargs)
        elif name == 'tradedays':
            df.to_excel(os.path.join(self.root, 'src', self.tradedays_file), encoding='gbk', **kwargs)
        else:
            path = self.freqmap.get(name, None)
            if path is None:
                path = self.root
            #if name in ['stm_issuingdate', 'applied_rpt_date_M']:
            #    df = df.replace(0, pd.NaT)
            df.to_csv(os.path.join(path, name+'.csv'), encoding='gbk', **kwargs)
            self.__update_frepmap()
        self.__update_attr(name)

#    @staticmethod
#    def _fill_nan(series, value=0, ffill=False):
#        if ffill:
#            series = series.fillna(method='ffill')
#        else:
#            if value:
#                start_valid_idx = np.where(pd.notna(series))[0][0]
#                series.loc[start_valid_idx:] = series.loc[start_valid_idx:].fillna(0)
#        return series

    def __update_attr(self, name):
        if name in self.__dict__:
            del self.__dict__[name]
        self.__dict__[name] = getattr(self, name, None)

    def __getattr__(self, name):
        if name not in self.__dict__:
            self.__dict__[name] = self.open_file(name)
        return self.__dict__[name]

class FactorGenerater:
    def __init__(self, using_fetch=False):
        self.data = Data()
        if not using_fetch:
            self.dates_d = sorted(self.adjfactor.columns)
            self.dates_m = sorted(self.pct_chg_M.columns)

    def __getattr__(self, name):
        return getattr(self.data, name, None)

    def _get_trade_days(self, startday, endday, freq=None):
        if freq is None:
            freq = self.freq
        startday, endday = pd.to_datetime((startday, endday))
        if freq == 'd':
            try:
                start_idx = self._get_date_idx(startday, self.tradedays)
            except IndexError:
                return []
            else:
                try:
                    end_idx = self._get_date_idx(endday, self.tradedays)
                except IndexError:
                    return self.tradedays[start_idx:]
                else:
                    return self.tradedays[start_idx:end_idx+1]
        else:
            new_cdays_curfreq = pd.Series(index=self.tradedays).resample(freq).asfreq().index
            c_to_t_dict = {cday:tday for tday, cday in self.month_map.to_dict().items()}
            try:
                new_tdays_curfreq = [c_to_t_dict[cday] for cday in new_cdays_curfreq]
            except KeyError:
                new_tdays_curfreq = [c_to_t_dict[cday] for cday in new_cdays_curfreq[:-1]]
            start_idx = self._get_date_idx(c_to_t_dict.get(startday, startday), new_tdays_curfreq) + 1
            try:
                end_idx = self._get_date_idx(c_to_t_dict.get(endday, endday), new_tdays_curfreq)
            except IndexError:
                end_idx = len(new_tdays_curfreq) - 1
            return new_tdays_curfreq[start_idx:end_idx+1]

    @lazyproperty
    def trade_days(self):
        self.__trade_days = self._get_trade_days(self.startday, self.endday)
        return self.__trade_days

    def save_file(self, datdf, path):
        datdf = datdf.loc[~pd.isnull(datdf['is_open1']), :]

        for col in ['name', 'industry_sw']:
            datdf[col] = datdf[col].apply(str)
        datdf = datdf.loc[~datdf['name'].str.contains('0')]

        save_cond1 = (~datdf['name'].str.contains('ST')) #剔除ST股票
        save_cond2 = (~pd.isnull(datdf['industry_sw'])) & (~datdf['industry_sw'].str.contains('0')) #剔除行业值为0或为空的股票
        save_cond3 = (~pd.isnull(datdf['MKT_CAP_FLOAT'])) #剔除市值为空的股票
        save_cond = save_cond1 & save_cond2 & save_cond3
        datdf = datdf.loc[save_cond]

        datdf = datdf.reset_index()
        datdf.index = range(1, len(datdf)+1)
        datdf.index.name = 'No'
        datdf = datdf.rename(columns={"index":"code"})

        #之前不管是计算指标还是计算因子,当某些除法操作分母为0的情况会导致产生inf值,所以这里统一处理
        datdf = datdf.replace(np.inf, 0).replace(-np.inf, 0)

        if path.endswith('.csv'):
            return datdf.to_csv(path, encoding='gbk')
        else:
            raise TypeError("Unsupportted type {}, only support csv currently.".format(path.split('.')[-1]))

    @staticmethod
    def concat_df(left, right, *, how="outer", left_index=True, right_index=True, **kwargs):
        return pd.merge(left, right, how=how, left_index=left_index, right_index=right_index, **kwargs)

    def create_factor_file(self, date, savepath):
        if os.path.exists(savepath):
            raise FileAlreadyExistError(f"{date}'s data already exist, please try calling update method.")
        stklist, dat0 = self.get_basic_data(date)
        dat1 = self.get_factor_data(date, stklist)
        res = self.concat_df(dat0, dat1)
        self.save_file(res, savepath)

    def get_basic_data(self, tdate):
        df0 = self.meta[self.meta['ipo_date'] <= tdate] #股票上市时间早于指定时间
        cond = (pd.isnull(df0['delist_date'])) | (df0['delist_date'] >= tdate) #股票退市时间晚于指定时间
        df0 = df0[cond]
        #接下来还需要判断如果每月停牌日期大于一定数目就排除这只股票
        bdate = self.trade_days_begin_end_of_month.at[tdate, 'month_start']
        tradestatus = self.trade_status.loc[df0.index, bdate:tdate]
        tradestatus = (tradestatus==0) #停牌的股票为True
        cond = (tradestatus.sum(axis=1) < 10) #停牌日期小于10天的股票才入选, 超过10天的排除
        df0 = df0[cond]
        df0 = df0.rename(columns={'sec_name':'name'})
        del df0['delist_date']
        stocklist = df0.index.tolist()
        caldate = self.month_map[tdate]
        df0["industry_zx"] = self.industry_citic.loc[stocklist, caldate] #中信行业分类
        df0["industry_sw"] = df0["industry_zx"]
        df0['MKT_CAP_FLOAT'] = self.mkt_cap_float_m.loc[stocklist, caldate]
        try:
            tdate = self._get_next_month_first_trade_date(tdate) #下个月第一个交易日
        except IndexError:
            df0["is_open1"] = None
            df0["PCT_CHG_NM"] = None
            return stocklist, df0
        df0["is_open1"] = self.trade_status.loc[stocklist, tdate].map({1:"TRUE", 0:"FALSE"})
        df0["PCT_CHG_NM"] = self.get_next_pctchg(stocklist, tdate) #下月的月收益率,回测的时候会使用到
        return stocklist, df0

    def _get_next_month_first_trade_date(self, date):
        date = pd.to_datetime(date)
        tdates = self.trade_status.columns.tolist()

        def _if_same_month(x):
            nonlocal date
            if date.month != 12:
                return (x.year != date.year) or (x.month - 1 != date.month)
            else:
                return (x.year - 1 != date.year) or (x.month != 1)

        daterange = dropwhile(_if_same_month, tdates)
        return list(daterange)[0]

    def get_next_pctchg(self, stocklist, tdate):
        try:
            nextdate = tdate + toffsets.MonthEnd(1)
            dat = self.pct_chg_M.loc[stocklist, nextdate]
        except Exception as e:
            print("Get next month data failed. msg: {}".format(e))
            dat = [np.nan] * len(stocklist)
        return dat

    def get_last_month_end(self, date):
        if date.month == 1:
            lstyear = date.year - 1
            lstmonth = 12
        else:
            lstyear = date.year
            lstmonth = date.month - 1
        return datetime(lstyear, lstmonth, 1) + toffsets.MonthEnd(n=1)

    def get_factor_data(self, tdate, stocklist):
        caldate = self.month_map[tdate]
        dat1 = self._get_value_data(stocklist, caldate)
        dat2 = self._get_growth_data(stocklist, caldate)
        dat3 = self._get_finance_data(stocklist, caldate)
        dat4 = self._get_leverage_data(stocklist, caldate)
        dat5 = self._get_cal_data(stocklist, tdate)
        dat6 = self._get_tech_data(stocklist, tdate)
        res = reduce(self.concat_df, [dat1, dat2, dat3, dat4, dat5, dat6])
        dat7 = self._get_barra_quote_data(stocklist, tdate)
        dat8 = self._get_barra_finance_data(stocklist, tdate)
        res = reduce(self.concat_df, [res, dat7, dat8])
        return res

    def _get_value_data(self, stocks, caldate):
        """
            Default value indicators getted from windpy:
            'pe_ttm', 'val_pe_deducted_ttm', 'pb_lf', 'ps_ttm', 
            'pcf_ncf_ttm', 'pcf_ocf_ttm', 'dividendyield2', 'profit_ttm'
            
            Default target value indicators:
            'EP', 'EPcut', 'BP', 'SP', 
            'NCFP', 'OCFP', 'DP', 'G/PE'
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat['EP'] = 1 / self.pe_ttm_m.loc[stocks, date]
        dat['EPcut'] = 1 / self.val_pe_deducted_ttm_m.loc[stocks, date]
        dat['BP'] = 1 / self.pb_lf_m.loc[stocks, date]
        dat['SP'] = 1 / self.ps_ttm_m.loc[stocks, date]
        dat['NCFP'] = 1 / self.pcf_ncf_ttm_m.loc[stocks, date]
        dat['OCFP'] = 1 / self.pcf_ocf_ttm_m.loc[stocks, date]
        dat['DP'] = self.dividendyield2_m.loc[stocks, date]
        dat['G/PE'] = self.profit_ttm_G_m.loc[stocks, date] * dat['EP']

        dat = dat[self.value_target_indicators]
        return dat

    def _get_growth_data(self, stocks, caldate):
        """
            Default growth indicators getted from windpy:
            "qfa_yoysales", "qfa_yoyprofit", "qfa_yoyocf", "qfa_roe"
            
            Default target growth indicators:
            "Sales_G_q","Profit_G_q", "OCF_G_q", "ROE_G_q", 
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["Sales_G_q"] = self.qfa_yoysales_m.loc[stocks, date]
        dat["Profit_G_q"] = self.qfa_yoyprofit_m.loc[stocks, date]
        dat["OCF_G_q"] = self.qfa_yoyocf_m.loc[stocks, date]
        dat['ROE_G_q'] = self.qfa_roe_G_m.loc[stocks, date]

        dat = dat[self.growth_target_indicators]
        return dat

    def _get_finance_data(self, stocks, caldate):
        """
            Default finance indicators getted from windpy:
            "roe_ttm2_m", "qfa_roe_m", 
            "roa2_ttm2_m", "qfa_roa_m", 
            "grossprofitmargin_ttm2_m", "qfa_grossprofitmargin_m", 
            "deductedprofit_ttm", "qfa_deductedprofit_m", "or_ttm", "qfa_oper_rev_m", 
            "turnover_ttm_m", "qfa_netprofitmargin_m", 
            "ocfps_ttm", "eps_ttm", "qfa_net_profit_is_m", "qfa_net_cash_flows_oper_act_m"
            
            Default target finance indicators:
            "ROE_q", "ROE_ttm", 
            "ROA_q", "ROA_ttm", 
            "grossprofitmargin_q", "grossprofitmargin_ttm", 
            "profitmargin_q", "profitmargin_ttm",
            "assetturnover_q", "assetturnover_ttm", 
            "operationcashflowratio_q", "operationcashflowratio_ttm"
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["ROE_q"] = self.qfa_roe_m.loc[stocks, date]
        dat["ROE_ttm"] = self.roe_ttm2_m.loc[stocks, date]

        dat["ROA_q"] = self.qfa_roa_m.loc[stocks, date]
        dat["ROA_ttm"] = self.roa2_ttm2_m.loc[stocks, date]

        dat["grossprofitmargin_q"] = self.qfa_grossprofitmargin_m.loc[stocks, date]
        dat["grossprofitmargin_ttm"] = self.grossprofitmargin_ttm2_m.loc[stocks, date]

        #dat["profitmargin_q"] = self.qfa_deductedprofit_m.loc[stocks, date] / self.qfa_oper_rev_m.loc[stocks, date]
        #dat["profitmargin_ttm"] = self.deductedprofit_ttm.loc[stocks, date] / self.or_ttm.loc[stocks, date]

        #dat["assetturnover_q"] = self.qfa_roa_m.loc[stocks, date] / self.qfa_netprofitmargin_m.loc[stocks, date]
        dat['assetturnover_ttm'] = self.turnover_ttm_m.loc[stocks, date]

        #dat["operationcashflowratio_q"] = self.qfa_net_cash_flows_oper_act_m.loc[stocks, date] / self.qfa_net_profit_is_m.loc[stocks, date]
        #dat["operationcashflowratio_ttm"] = self.ocfps_ttm.loc[stocks, date] / self.eps_ttm.loc[stocks, date]

        #dat = dat[self.finance_target_indicators]
        return dat

    def _get_leverage_data(self, stocks, caldate):
        """
            Default leverage indicators getted from windpy:
            "assetstoequity_m", "longdebttoequity_m", "cashtocurrentdebt_m", "current_m"
            
            Default target leverage indicators:
            "financial_leverage", "debtequityratio", "cashratio", "currentratio"
        """
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat["financial_leverage"] = self.assetstoequity_m.loc[stocks, date]
        dat["debtequityratio"] = self.longdebttoequity_m.loc[stocks, date]
        dat["cashratio"] = self.cashtocurrentdebt_m.loc[stocks, date]
        dat["currentratio"] = self.current_m.loc[stocks, date]

        dat = dat[self.leverage_target_indicators]
        return dat

    def _get_cal_data(self, stocks, tdate):
        """
            Default calculated indicators getted from windpy:
            "mkt_cap_float", "holder_avgpct", "holder_num"
            
            Default target calculated indicators:
            "ln_capital", 
            "HAlpha", 
            "return_1m", "return_3m", "return_6m", "return_12m", 
            "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
            "exp_wgt_return_1m",  "exp_wgt_return_3m",  "exp_wgt_return_6m", "exp_wgt_return_12m", 
            "std_1m", "std_3m", "std_6m", "std_12m", 
            "beta", 
            "turn_1m", "turn_3m", "turn_6m", "turn_12m", 
            "bias_turn_1m", "bias_turn_3m", "bias_turn_6m", "bias_turn_12m", 
            "holder_avgpctchange"
        """
        tdate = pd.to_datetime(tdate)
        dat = pd.DataFrame(index=stocks)

        caldate = self.month_map[tdate]

        dat['ln_capital'] = np.log(self.mkt_cap_float_m.loc[stocks, caldate])
        #dat['holder_avgpctchange'] = self.holder_avgpctchg.loc[stocks, caldate]

        dat1 = self._get_mom_vol_data(stocks, tdate, self.dates_d, params=[1,3,6,12])
        dat2 = self._get_turnover_data(stocks, tdate, self.dates_d, params=[1,3,6,12])
        dat3 = self._get_regress_data(stocks, tdate, self.dates_m, params=["000001.SH", 24])

        dat = reduce(self.concat_df, [dat, dat1, dat2, dat3])
        #dat = dat[self.cal_target_indicators]
        return dat

    def _get_tech_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "close(freq=d)"
            
            Default target technique indicators:
            "MACD", "DEA", "DIF", "RSI", "PSY", "BIAS"
        """
        dat = pd.DataFrame(index=stocks)
        for tname in self.tech_indicators:
            calfunc = getattr(self, 'cal_'+tname, None)
            if calfunc is None:
                msg = "Please define property:'{}' first.".format("cal_"+tname)
                raise NotImplementedError(msg)
            else:
                if tname == "MACD":
                    dat["DIF"], dat["DEA"], dat["MACD"] = calfunc(stocks, tdate, self._tech_params[tname])
                else:
                    dat[tname] = calfunc(stocks, tdate, self._tech_params[tname])
        return dat

    def _get_mom_vol_data(self, stocks, tdate, dates, params=(1,3,6,12)):
        pct_chg = self.pct_chg
        turnover = self.turn
        caldate = self.month_map[tdate]
        res = pd.DataFrame(index=stocks)
        for offset in params:
            period_d = self._get_period_d(tdate, offset=-offset, freq="M", datelist=dates)
            cur_pct_chg_d = pct_chg.loc[stocks, period_d]
            cur_turnover = turnover.loc[stocks, period_d]
            wgt_pct_chg = cur_pct_chg_d * cur_turnover
            days_wgt = cur_pct_chg_d.expanding(axis=1).apply(lambda df: np.exp(-(len(period_d) - len(df))/4/offset))
            exp_wgt_pct_chg = wgt_pct_chg * days_wgt
            cur_pct_chg_m = getattr(self, f"pctchg_{offset}M", None)
            res[f"return_{offset}m"] = cur_pct_chg_m.loc[stocks, caldate]
            res[f"wgt_return_{offset}m"] = wgt_pct_chg.apply(np.nanmean, axis=1)
            res[f"exp_wgt_return_{offset}m"] = exp_wgt_pct_chg.apply(np.nanmean, axis=1)
            res[f"std_{offset}m"] = cur_pct_chg_d.apply(np.nanstd, axis=1)
        return res

    def _get_turnover_data(self, stocks, tdate, dates, params=(1,3,6,12)):
        base_period_d = self._get_period_d(tdate, offset=-2, freq="y", datelist=dates)
        cur_turnover_base = self.turn.loc[stocks, base_period_d]
        turnover_davg_base = cur_turnover_base.apply(np.nanmean, axis=1)
        res = pd.DataFrame(index=stocks)
        for offset in params:
            period_d = self._get_period_d(tdate, offset=-offset, freq="M", datelist=dates)
            cur_turnover = self.turn.loc[stocks, period_d]
            turnover_davg = cur_turnover.apply(np.nanmean, axis=1)
            res[f"turn_{offset}m"] = turnover_davg
            res[f"bias_turn_{offset}m"] = turnover_davg / turnover_davg_base - 1
        return res

    def _get_regress_data(self, stocks, tdate, dates, params=("000001.SH", 60)):
        """
            return value contains:
            HAlpha --intercept
            beta   --slope
        """
        index_code, period = params

        col_index = self._get_period(tdate, offset=-period, freq="M", datelist=dates, resample=False) #前推60个月(五年)
        pct_chg_idx = self.pct_chg_M.loc[index_code, col_index]
        pct_chg_m = self.pct_chg_M.loc[stocks, col_index].dropna(how='any', axis=0).T
        x, y = pct_chg_idx.values.reshape(-1,1), pct_chg_m.values

        valid_stocks = pct_chg_m.columns.tolist()
        try:
            beta, Halpha = self.regress(x, y)
        except ValueError as e:
            print(e)
            #raise
            beta, Halpha = np.empty((len(valid_stocks),1)), np.empty((1, len(valid_stocks)))

        beta = pd.DataFrame(beta, index=valid_stocks, columns=['beta'])
        Halpha = pd.DataFrame(Halpha.T, index=valid_stocks, columns=['HAlpha'])
        res = self.concat_df(beta, Halpha)
        return res

    def _get_barra_quote_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "mkt_cap_float", "pct_chg", "amt"
            
            Default target barra_quote indicators:
            "LNCAP_barra", "MIDCAP_barra", 
            "BETA_barra", "HSIGMA_barra", "HALPHA_barra",
            "DASTD_barra", "CMRA_barra",
            "STOM_barra", "STOQ_barra", "STOA_barra",
            "RSTR_barra"
        """
        tdate = pd.to_datetime(tdate)
        caldate = self.month_map[tdate]
        dat = pd.DataFrame(index=stocks)

        dat1 = self._get_size_barra(stocks, caldate, self.dates_d, params=[True,True,True])
        dat2 = self._get_regress_barra(stocks, tdate, self.dates_d, params=[4,504,252,True,'000300.SH'])
        dat3 = self._get_dastd_barra(stocks, tdate, self.dates_d, params=[252,42])
        dat4 = self._get_cmra_barra(stocks, tdate, self.dates_d, params=[12, 21])
        dat5 = self._get_liquidity_barra(stocks, tdate, params=[21,1,3,12])
        dat6 = self._get_rstr_barra(stocks, tdate, self.dates_d, params=[252,126,11,'000300.SH'])

        dat = reduce(self.concat_df, [dat, dat1, dat2, dat3, dat4, dat5, dat6])
        dat = dat[self.barra_quote_target_indicators]
        return dat

    def _get_size_barra(self, stocks, caldate, dates, params=(True,True,True)):
        intercept, standardize, wls = params

        res = pd.DataFrame(index=stocks)
        lncap = self.mkt_cap_float_m.loc[stocks, caldate].apply(np.log)
        lncap_3 = lncap ** 3

        if wls:
            w = lncap.apply(np.sqrt)
            x_y_w = pd.concat([lncap, lncap_3, w], axis=1).dropna(how='any', axis=0)
            x, y, w = x_y_w.iloc[:,0], x_y_w.iloc[:,1], x_y_w.iloc[:,-1]
            x, y, w = x.values, y.values, w.values
        else:
            w = 1
            x_and_y = pd.concat([lncap, lncap_3], axis=1).dropna(how='any', axis=0)
            x, y = x_and_y.iloc[:,0], x_and_y.iloc[:,-1]
            x, y = x.values, y.values

        intercept, coef = self.regress(x, y, intercept, w)
        resid = lncap_3 - (coef * lncap + intercept)

        if standardize:
            resid = self.standardize(self.winsorize(resid))
        res['MIDCAP_barra'] = resid
        res['LNCAP_barra'] = lncap
        return res

    def _get_regress_barra(self, stocks, tdate, dates_d, params=(4,504,252,True,'000300.SH')):
        shift, window, half_life, if_intercept, index_code = params
        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life)
        idx = self._get_date_idx(tdate, dates_d)
        date_period = dates_d[idx-window+1-shift:idx+1]
        pct_chgs = self.pct_chg.T.loc[date_period,:]

        for i in range(1,shift+1):
            pct_chg = pct_chgs.iloc[i:i+window,:]
            x = pct_chg.loc[:, index_code]
            ys = pct_chg.loc[:, stocks].dropna(how='any', axis=1)
            X, Ys = x.values, ys.values
            try:
                intercept, coef = self.regress(X, Ys, if_intercept, w)
            except:
                print(X)
                print(Ys)
                raise
            alpha = pd.Series(intercept, index=ys.columns)
            beta = pd.Series(coef[0], index=ys.columns)
            alpha.name = f'alpha_{i}'; beta.name = f'beta_{i}'
            res = pd.concat([res, alpha, beta], axis=1)
            if i == shift:
                resid = Ys - (intercept + X.reshape(-1,1) @ coef)
                sigma = pd.Series(np.std(resid, axis=0), index=ys.columns)
                sigma.name = 'HSIGMA_barra'
                res = pd.concat([res, sigma], axis=1)

        res['HALPHA_barra'] = np.sum((res[f'alpha_{i}'] for i in range(1,shift+1)), axis=0)
        res['BETA_barra'] = np.sum((res[f'beta_{i}'] for i in range(1,shift+1)), axis=0)
        res = res[['BETA_barra', 'HALPHA_barra', 'HSIGMA_barra']]
        return res

    def _get_dastd_barra(self, stocks, tdate, dates_d, params=(252,42)):
        window, half_life = params

        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life)
        pct_chg = self._get_daily_data("pct_chg", stocks, tdate, window, dates_d)
        pct_chg = pct_chg.dropna(how='any', axis=1)
        res['DASTD_barra'] = pct_chg.apply(self._std_dev, args=(w,))
        return res

    @staticmethod
    def _std_dev(series, weight=1):
        mean = np.mean(series)
        std_dev = np.sqrt(np.sum((series - mean)**2 * weight))
        return std_dev

    def _get_cmra_barra(self, stocks, tdate, dates_d, params=(12,21)):
        months, days_pm = params
        window = months * days_pm

        res = pd.DataFrame(index=stocks)
        pct_chg = self._get_daily_data("pct_chg", stocks, tdate, window, dates_d)
        pct_chg = pct_chg.dropna(how='any', axis=1)
        res['CMRA_barra'] = np.log(1 + pct_chg).apply(self._cal_cmra, args=(months, days_pm))
        return res

    @staticmethod
    def _cal_cmra(series, months=12, days_per_month=21):
        z = sorted(series[-i * days_per_month:].sum() for i in range(1, months+1))
        return z[-1] - z[0]

    def _get_liquidity_barra(self, stocks, tdate, params=(21,1,3,12)):
        days_pm, freq1, freq2, freq3 = params
        window = freq3 * days_pm

        res = pd.DataFrame(index=stocks)
        amt = self._get_daily_data('amt', stocks, tdate, window)
        mkt_cap_float = self._get_daily_data('mkt_cap_float', stocks, tdate, window)
        share_turnover = amt / mkt_cap_float

        for freq in [freq1, freq2, freq3]:
            res[f'st_{freq}'] = share_turnover.iloc[-freq*days_pm:,:].apply(self._cal_liquidity, args=(freq,))
        res = res.rename(columns={f'st_{freq1}':'STOM_barra',
                                  f'st_{freq2}':'STOQ_barra',
                                  f'st_{freq3}':'STOA_barra'})
        return res

    @staticmethod
    def _cal_liquidity(series, freq=1):
        res = np.log(np.nansum(series) / freq)
        return np.where(np.isinf(res), 0, res)

    def _get_rstr_barra(self, stocks, tdate, dates_d, params=(252,126,11,'000300.SH')):
        window, half_life, shift, index_code = params

        res = pd.DataFrame(index=stocks)
        w = self.get_exponential_weights(window, half_life)
        idx = self._get_date_idx(tdate, dates_d)
        date_period = dates_d[idx-window-shift+1:idx+1]
        pct_chgs = self.pct_chg.T.loc[date_period, :]

        for i in range(1,shift+1):
            pct_chg = pct_chgs.iloc[i:i+window,:]
            stk_ret = pct_chg[stocks]
            bm_ret = pct_chg[index_code]
            excess_ret = np.log(1 + stk_ret).sub(np.log(1 + bm_ret), axis=0)
            excess_ret = excess_ret.mul(w, axis=0)
            rs = excess_ret.apply(np.nansum, axis=0)
            rs.name = f'rs_{i}'
            res = pd.concat([res, rs], axis=1)

        res['RSTR_barra'] = np.sum((res[f'rs_{i}'] for i in range(1,shift+1)), axis=0) / shift
        return res[['RSTR_barra']]

    def _get_barra_finance_data(self, stocks, tdate):
        """
            Default source data loaded from local file:
            "mkt_cap_ard", "longdebttodebt", "other_equity_instruments_PRE", 
            "tot_equity", "tot_liab", "tot_assets", "pb_lf", 
            "pe_ttm", "pcf_ocf_ttm", "eps_diluted2", "orps"
            
            Default target barra_quote indicators:
            "MLEV_barra", "BLEV_barra", "DTOA_barra", "BTOP_barra", 
            "ETOP_barra", "CETOP_barra", "EGRO_barra", "SGRO_barra"
        """
        dat = pd.DataFrame(index=stocks)
        caldate = self.month_map[tdate]

        dat1 = self._get_leverage_barra(stocks, tdate, self.dates_d)
        dat2 = self._get_value_barra(stocks, caldate)
        #dat3 = self._get_growth_barra(stocks, caldate, params=(5,'y'))

        dat = reduce(self.concat_df, [dat, dat1, dat2, ])
        #dat = dat[self.barra_finance_target_indicators]
        return dat

    def _get_leverage_barra(self, stocks, tdate, dates):
        lst_tdate = self._get_date(tdate, -1, dates)
        caldate = self.month_map[tdate]
        dat = pd.DataFrame(index=stocks)
        try:
            long_term_debt = self.longdebttodebt_lyr_m.loc[stocks, caldate] * self.tot_liab_lyr_m.loc[stocks, caldate]
        except Exception:
            print(caldate, len(stocks))
            raise

        prefered_equity = self.other_equity_instruments_PRE_lyr_m.loc[stocks, caldate].fillna(0)

        dat['MLEV_barra'] = (prefered_equity + long_term_debt) / (self.mkt_cap_ard.loc[stocks, lst_tdate]) + 1
        dat['BLEV_barra'] = (self.tot_equity_lyr_m.loc[stocks, caldate] + long_term_debt) / (self.tot_equity_lyr_m.loc[stocks, caldate] - prefered_equity)
        dat['DTOA_barra'] = self.tot_liab_lyr_m.loc[stocks, caldate] / self.tot_assets_lyr_m.loc[stocks, caldate]
        return dat

    def _get_value_barra(self, stocks, caldate):
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        dat['BTOP_barra'] = 1 / self.pb_lf_m.loc[stocks, date]
        dat['ETOP_barra'] = 1 / self.pe_ttm_m.loc[stocks, date]
        dat['CETOP_barra'] = 1 / self.pcf_ocf_ttm_m.loc[stocks, date]
        return dat

    def _get_growth_barra(self, stocks, caldate, params=(5, 'y')):
        periods, freq = params
        date = pd.to_datetime(caldate)
        dat = pd.DataFrame(index=stocks)

        eps = self.eps_diluted2.loc[stocks,:]
        orps = self.orps.loc[stocks,:]
        dat['EGRO_barra'] = self._cal_growth_rate(eps, stocks, date, periods, freq)
        dat['SGRO_barra'] = self._cal_growth_rate(orps, stocks, date, periods, freq)
        return dat

    @staticmethod
    def _get_lyr_date(date):
        if date.month == 12:
            return date
        else:
            try:
                return pd.to_datetime(f'{date.year-1}-12-31')
            except:
                return pd.NaT

    def __cal_gr(self, series, lyr_rptdates, periods=5):
        lyr_date = lyr_rptdates[series.name]
        if pd.isna(lyr_date):
            return np.nan
        idx = self._get_date_idx(lyr_date, series.index)
        y = series.iloc[idx-periods+1:idx+1]
        x = pd.Series(range(1, len(y)+1), index=y.index)

        x_and_y = pd.concat([x,y], axis=1).dropna(how='any', axis=1)
        try:
            x, y = x_and_y.iloc[:, 0].values, x_and_y.iloc[:, 1].values
            _, coef = self.regress(x,y)
            return coef[0] / np.mean(y)
        except:
            return np.nan

    def _cal_growth_rate(self, ori_data, stocks, caldate, periods=5, freq='y'):
        try:
            current_rptdates = self.applied_rpt_date_M.loc[stocks, caldate]
        except Exception:
            print(stocks[:5])
            print(caldate)
            print(type(stocks), type(caldate))
            raise
        current_lyr_rptdates = current_rptdates.apply(self._get_lyr_date)
        #tdate = pd.to_datetime('2019-03-29'); self = z; caldate = self.month_map[tdate]
        #stocks = self._FactorProcess__get_stock_list(tdate); ori_data = self.current.loc[stocks,:]
        if ori_data.index.dtype == 'O':
            ori_data = ori_data.T
        ori_data = ori_data.groupby(pd.Grouper(freq=freq)).apply(lambda df: df.iloc[-1])
        res = ori_data.apply(self.__cal_gr, args=(current_lyr_rptdates, periods))
        return res

    @staticmethod
    def get_exponential_weights(window=12, half_life=6):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1]

    @staticmethod
    def winsorize(dat, n=5):
        dm = np.nanmedian(dat, axis=0)
        dm1 = np.nanmedian(np.abs(dat - dm), axis=0)
        if len(dat.shape) > 1:
            dm = np.repeat(dm.reshape(1,-1), dat.shape[0], axis=0)
            dm1 = np.repeat(dm1.reshape(1,-1), dat.shape[0], axis=0)
        dat = np.where(dat > dm + n * dm1, dm + n * dm1, 
              np.where(dat < dm - n * dm1, dm - n * dm1, dat))
        return dat

    @staticmethod
    def standardize(dat):
        dat_sta = (dat - np.nanmean(dat, axis=0)) / np.nanstd(dat, axis=0)
        return dat_sta

    @staticmethod
    def regress(X, y, intercept=True, weights=1, robust=False):
        if intercept:
            X = sm.add_constant(X)
        if robust:
            model = sm.RLM(y, X, weights=weights)
        else:
            model = sm.WLS(y, X, weights=weights)
        result = model.fit()
        params = result.params 
        return params[0], params[1:]

    @staticmethod
    def get_sma(df, n, m):
        try:
            sma = pd.ewma(df, com=n/m-1, adjust=False, ignore_na=True)
        except AttributeError:
            sma = df.ewm(com=n/m-1, min_periods=0, adjust=False, ignore_na=True).mean()
        return sma

    @staticmethod
    def get_ema(df, n):
        try:
            ema = pd.ewma(df, span=n, adjust=False, ignore_na=True)
        except AttributeError:
            ema = df.ewm(span=n, min_periods=0, adjust=False, ignore_na=True).mean()
        return ema

    def _get_daily_data(self, name, stocks, date, offset, datelist=None):
        dat = getattr(self, name, None)
        if dat is None:
            raise AttributeError("{} object has no attr: {}".format(self.__class__.__name__, name))

        dat = dat.loc[stocks, :].T
        if datelist is None:
            datelist = dat.index.tolist()
        idx = self._get_date_idx(date, datelist)
        start_idx, end_idx = max(idx-offset+1, 0), idx+1
        date_period = datelist[start_idx:end_idx]
        dat = dat.loc[date_period, :]
        return dat

    def cal_MACD(self, stocks, date, params=(12,26,9)):
        n1, n2, m = params
        offset = max([n1,n2,m]) + 240
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        dif = self.get_ema(close, n1) - self.get_ema(close, n2)
        dea = self.get_ema(dif, m)
        macd = 2*(dif - dea)

        dif = dif.iloc[-1, :].T.values
        dea = dea.iloc[-1, :].T.values
        macd = macd.iloc[-1, :].T.values
        return dif, dea, macd

    def cal_PSY(self, stocks, date, params=(20,)):
        m = params[0]
        offset = m + 1
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        con = (close > close.shift(1)).astype(int)
        psy = 100 * con.rolling(window=m).sum() / m

        return psy.iloc[-1, :].T.values

    def cal_RSI(self, stocks, date, params=(20,)):
        n = params[0]
        offset = n + 1
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        delta = close - close.shift(1)
        tmp1 = delta.where(delta > 0, 0)
        tmp2 = delta.applymap(abs)
        rsi = 100 * self.get_sma(tmp1, n, 1) / self.get_sma(tmp2, n, 1)

        return rsi.iloc[-1, :].T.values

    def cal_BIAS(self, stocks, date, params=(20,)):
        n = params[0]
        offset = n
        close = self._get_daily_data("hfq_close", stocks, date, offset)

        ma_close = close.rolling(window=n).mean()
        bias = 100 * (close - ma_close) / ma_close

        return bias.iloc[-1, :].T.values

    def _get_date_idx(self, date, datelist=None, ensurein=False):
        msg = """Date {} not in current tradedays list. If this date value is certainly a tradeday,  
              please reset tradedays list with longer periods or higher frequency."""
        date = pd.to_datetime(date)
        if datelist is None:
            datelist = self.trade_days
        try:
            datelist = sorted(datelist)
            idx = datelist.index(date)
        except ValueError:
            if ensurein:
                raise IndexError(msg.format(str(date)[:10]))
            dlist = list(datelist)
            dlist.append(date)
            dlist.sort()
            idx = dlist.index(date) 
            if idx == len(dlist)-1:
                raise IndexError(msg.format(str(date)[:10]))
            return idx - 1
        return idx

    def _get_date(self, date, offset=0, datelist=None):
        if datelist is None:
            datelist = self.trade_days
        try:
            idx = self._get_date_idx(date, datelist)
        except IndexError as e:
            print(e)
            idx = len(datelist) - 1
        finally:
            return datelist[idx+offset]

    def _get_period_d(self, date, offset=None, freq=None, datelist=None):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")

        conds = {}
        freq = freq.upper()
        if freq == "M":
            conds.update(months=-offset)
        elif freq == "Q":
            conds.update(months=-3*offset)
        elif freq == "Y":
            conds.update(years=-offset)
        else:
            freq = freq.lower()
            conds.update(freq=-offset)

        start_date = pd.to_datetime(date) - pd.DateOffset(**conds)

        if start_date.month == 12:
            year = start_date.year + 1
            month = 1
        else:
            year = start_date.year
            month = start_date.month + 1
        day = 1

        sdate = datetime(year, month, day)

        if datelist is None:
            datelist = self.dates_d
        try:
            sindex = self._get_date_idx(sdate, datelist, ensurein=True)
            eindex = self._get_date_idx(date, datelist, ensurein=True)
            return datelist[sindex:eindex+1]
        except IndexError:
            return self._get_trade_days(sdate, date, "d")

    def _get_period(self, date, offset=-1, freq=None, datelist=None, resample=False):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")

        date = pd.to_datetime(date)
        if resample:
            if datelist:
                datelist = self._transfer_freq(datelist, freq)
            else:
                raise ValueError("Can resample on passed in datelist.")

        if freq is None or freq == self.freq:
            if datelist:
                end_idx = self._get_date_idx(date, datelist) + 1
            else:
                end_idx = self._get_date_idx(date) + 1
        else:
            if datelist:
                end_idx = self._get_date_idx(date, datelist) + 1
            else:
                msg = """Must pass in a datelist with freq={} since it is not conformed with default freq."""
                raise ValueError(msg.format(freq))
        start_idx = end_idx + offset
        return datelist[start_idx: end_idx]

    def _transfer_freq(self, daylist=None, freq='M'):
        if daylist is None:
            daylist = self.pct_chg_M.columns.tolist()
        freq = freq.upper()
        if freq == "M":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month)
        elif freq == "Q":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month and lst.month in (3,6,9,12))
        elif freq == "Y":
            res = (lst for lst, td in zip(daylist[:-1], daylist[1:]) if lst.month != td.month and lst.month == 12)
        else:
            raise TypeError("Unsupported resample type {}.".format(freq))
        return list(res)
