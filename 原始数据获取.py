# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
from joblib import Parallel, delayed
from raw_data_fetch import TushareFetcher, WindFetcher

#---------------------------------------------------------------
# Tushare数据源
#---------------------------------------------------------------
def TushareFetch():
    fetcher = TushareFetcher()
    #---------------------------------------------------------------
    # 先下载数据到本地
    #---------------------------------------------------------------
    fetcher.fetch_meta_data()
    fetcher.fetch_trade_day()
    fetcher.fetch_month_map()
    fetcher.ensure_data(fetcher.daily, "__temp_daily__") #日行情表
    fetcher.ensure_data(fetcher.suspend_d, "__temp_suspend_d__") #停牌表
    fetcher.ensure_data(fetcher.limit_list, "__temp_limit_list__") #涨跌停表
    fetcher.ensure_data(fetcher.adj_factor, "__temp_adj_factor__") #复权因子表
    fetcher.ensure_data(fetcher.daily_basic, "__temp_daily_basic__") #每日指标表
    fetcher.ensure_data(fetcher.moneyflow, "__temp_moneyflow__") #资金流表
    fetcher.ensure_data_by_q(fetcher.fina_indicator, "__temp_fina_indicator__") #财务指标表
    fetcher.ensure_data_by_q(fetcher.income, "__temp_income__") #利润表
    fetcher.ensure_data_by_q(fetcher.balancesheet, "__temp_balancesheet__") #资产负债表
    fetcher.ensure_data_by_q(fetcher.cashflow, "__temp_cashflow__") #现金流表
    fetcher.index_daily()
    #---------------------------------------------------------------
    # 然后从本地数据生成指标
    #---------------------------------------------------------------
    fetcher.create_listday_matrix()
    fetcher.create_month_tdays_begin_end()
    fetcher.create_turn_d()
    fetcher.create_trade_status()
    fetcher.create_maxupordown()
    fetcher.create_indicator("__temp_adj_factor__", "adj_factor", "adjfactor")
    fetcher.create_mkt_cap_float_m()
    fetcher.create_pe_ttm_m()
    fetcher.create_val_pe_deducted_ttm_m()
    fetcher.create_pb_lf_m()
    fetcher.create_ps_ttm_m()
    fetcher.create_pcf_ncf_ttm_m()
    fetcher.create_pcf_ocf_ttm_m()
    fetcher.create_dividendyield2_m()
    fetcher.create_profit_ttm_G_m()
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_sales_yoy", "qfa_yoysales_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_profit_yoy", "qfa_yoyprofit_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "ocf_yoy", "qfa_yoyocf_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yoy", "qfa_roe_G_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_roe", "qfa_roe_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yearly", "roe_ttm2_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa", "qfa_roa_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa_yearly", "roa2_ttm2_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_gsprofit_margin", "qfa_grossprofitmargin_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "grossprofit_margin", "grossprofitmargin_ttm2_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_turn", "turnover_ttm_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_to_eqt", "assetstoequity_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "debt_to_eqt", "longdebttoequity_m") #临时替代
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "cash_to_liqdebt", "cashtocurrentdebt_m")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "current_ratio", "current_m")
    fetcher.create_daily_quote_indicators()
    fetcher.create_indicator("__temp_daily_basic__", "circ_mv", "mkt_cap_float")
    fetcher.create_indicator("__temp_daily_basic__", "total_mv", "mkt_cap_ard")
    fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "longdeb_to_debt", "longdebttodebt_lyr_m")
    fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_liab", "tot_liab_lyr_m")
    fetcher.create_indicator_m_by_q("__temp_balancesheet__", "oth_eqt_tools_p_shr", "other_equity_instruments_PRE_lyr_m")
    fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_hldr_eqy_inc_min_int", "tot_equity_lyr_m")
    fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_assets", "tot_assets_lyr_m")


#---------------------------------------------------------------
# Wind数据源
#---------------------------------------------------------------
fetcher = WindFetcher()

def profit_ttm_G_m(): #净利润(ttm)同比增长率
    fetcher.create_profit_ttm_G_m()

def qfa_yoysales_m(): #营业收入(单季同比%)/成长
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_YOYSALES", "qfa_yoysales_m")

def qfa_yoyprofit_m(): #净利润(单季同比%)/成长
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_YOYPROFIT", "qfa_yoyprofit_m")

def qfa_yoyocf_m(): #经营现金流(单季同比%)/成长
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_YOYOCF", "qfa_yoyocf_m")

def qfa_roe_G_m(): #ROE(单季)同比增长率/成长
    fetcher.create_qfa_roe_G_m()

def roe_ttm2_m(): #ROE_ttm/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_ROE_TTM", "roe_ttm2_m")

def qfa_roa_m(): #ROA(单季)/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_ROA", "qfa_roa_m")

def roa2_ttm2_m(): #ROA_ttm/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_ROA2_TTM", "roa2_ttm2_m")

def qfa_grossprofitmargin_m(): #毛利率(单季)/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_GROSSPROFITMARGIN", "qfa_grossprofitmargin_m")

def grossprofitmargin_ttm2_m(): #毛利率(ttm)/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_GROSSPROFITMARGIN_TTM", "grossprofitmargin_ttm2_m")

def turnover_ttm_m(): #总资产周转率(ttm)/财务质量
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_ASSETSTURN", "turnover_ttm_m") #临时替代

def assetstoequity_m(): #权益乘数/杠杆
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_ASSETSTOEQUITY", "assetstoequity_m")

def longdebttoequity_m(): #非流动负债权益比/杠杆
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_DEBTTOEQUITY", "longdebttoequity_m") #临时替代

def cashtocurrentdebt_m(): #现金比率/杠杆
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_CASHTOLIQDEBT", "cashtocurrentdebt_m")

def current_m(): #流动比率/杠杆
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_CURRENT", "current_m")

def longdebttodebt_lyr_m():
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_LONGDEBTODEBT", "longdebttodebt_lyr_m")

def tot_liab_lyr_m():
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_LIAB", "tot_liab_lyr_m")

def other_equity_instruments_PRE_lyr_m():
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "OTHER_EQUITY_TOOLS_P_SHR", "other_equity_instruments_PRE_lyr_m")

def tot_equity_lyr_m():
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_SHRHLDR_EQY_INCL_MIN_INT", "tot_equity_lyr_m")

def tot_assets_lyr_m():
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_ASSETS", "tot_assets_lyr_m")

def WindFetch():

    #---------------------------------------------------------------
    # 先下载数据到本地
    #---------------------------------------------------------------
    #fetcher.ensure_data(fetcher.daily, "__temp_daily__") #日行情表
    #fetcher.ensure_data(fetcher.daily_basic, "__temp_daily_basic__") #每日指标表
    #fetcher.ensure_data_by_q(fetcher.fina_indicator, "__temp_fina_indicator__") #财务指标表
    #fetcher.ensure_data_by_q(fetcher.fina_indicator_ttm, "__temp_fina_indicator_ttm__") #财务指标(TTM)表
    #fetcher.ensure_data_by_q(fetcher.income, "__temp_income__") #利润表
    #fetcher.ensure_data_by_q(fetcher.balancesheet, "__temp_balancesheet__") #资产负债表
    #fetcher.ensure_data_by_q(fetcher.cashflow, "__temp_cashflow__") #现金流表
    #fetcher.ensure_data(fetcher.suspend_d, "__temp_suspend_d__") #停牌表

    #---------------------------------------------------------------
    # 然后从本地数据生成指标
    #---------------------------------------------------------------

    #并行方式
    function_list = [
        delayed(profit_ttm_G_m)(),
        delayed(qfa_yoysales_m)(),
        delayed(qfa_yoyprofit_m)(),
        delayed(qfa_yoyocf_m)(),
        delayed(qfa_roe_G_m)(),
        delayed(roe_ttm2_m)(),
        delayed(qfa_roa_m)(),
        delayed(roa2_ttm2_m)(),
        delayed(qfa_grossprofitmargin_m)(),
        delayed(grossprofitmargin_ttm2_m)(),
        delayed(turnover_ttm_m)(),
        delayed(assetstoequity_m)(),
        delayed(longdebttoequity_m)(),
        delayed(cashtocurrentdebt_m)(),
        delayed(current_m)(),
        delayed(longdebttodebt_lyr_m)(),
        delayed(tot_liab_lyr_m)(),
        delayed(other_equity_instruments_PRE_lyr_m)(),
        delayed(tot_equity_lyr_m)(),
        delayed(tot_assets_lyr_m)(),
    ]
    Parallel(n_jobs=10, backend='multiprocessing')(function_list) #并行化处理

    #串行方式
    #fetcher.create_trade_status()
    '''
    通过日频行情数据创建日频指标
    空值填充方式: 先ffill 后bfill
    '''
    #fetcher.create_daily_quote_indicators()
    '''
    通过日频指标数据创建日频指标
    空值填充方式: 先ffill 后bfill
    '''
    fetcher.create_daily_basic_indicators()
    '''
    通过日频指标数据创建月频指标
    空值填充方式: 先ffill 后bfill
    备注:最合理的填充方式应该是先把基础日频数据进行空值填充,然后再进行月频采样,而不是先月频采样再进行空值填充
    '''
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "circ_mv", "mkt_cap_float")
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "pe_ttm", "pe_ttm")
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "pe", "val_pe_deducted_ttm")
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "pb", "pb_lf")
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "ps_ttm", "ps_ttm")
    fetcher.create_indicator_m_by_d_ex("__temp_daily_basic__", "dv_ttm", "dividendyield2")
    '''
    通过季频财务数据创建月频指标
    空值填充方式: 先ffill 后bfill
    '''
    fetcher.create_profit_ttm_G_m()
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_YOYSALES", "qfa_yoysales_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_YOYPROFIT", "qfa_yoyprofit_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_YOYOCF", "qfa_yoyocf_m")
    fetcher.create_qfa_roe_G_m()
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_ROE_TTM", "roe_ttm2_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_ROA", "qfa_roa_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_ROA_TTM", "roa2_ttm2_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_QFA_GROSSPROFITMARGIN", "qfa_grossprofitmargin_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator_ttm__", "S_FA_GROSSPROFITMARGIN_TTM", "grossprofitmargin_ttm2_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_ASSETSTURN", "turnover_ttm_m") #临时替代
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_ASSETSTOEQUITY", "assetstoequity_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_DEBTTOEQUITY", "longdebttoequity_m") #临时替代
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_CASHTOLIQDEBT", "cashtocurrentdebt_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_CURRENT", "current_m")
    fetcher.create_indicator_m_by_q_ex("__temp_fina_indicator__", "S_FA_LONGDEBTODEBT", "longdebttodebt_lyr_m")
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_LIAB", "tot_liab_lyr_m")
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "OTHER_EQUITY_TOOLS_P_SHR", "other_equity_instruments_PRE_lyr_m")
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_SHRHLDR_EQY_INCL_MIN_INT", "tot_equity_lyr_m")
    fetcher.create_indicator_m_by_q_ex("__temp_balancesheet__", "TOT_ASSETS", "tot_assets_lyr_m")


if __name__ == '__main__':
    WindFetch()
    #TushareFetch()

    '''
    path = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(path, "industry_zx.csv"), index_col=[0], engine='python', encoding='gbk')
    df.columns = pd.to_datetime(df.columns)
    new_df = pd.DataFrame(index=df.index)
    def _get_month_end(date):
        import calendar
        import pandas.tseries.offsets as toffsets
        _, days = calendar.monthrange(date.year, date.month)
        if date.day == days:
            return date
        else:
            return date + toffsets.MonthEnd(n=1)
    for tday in df.columns.tolist():
        cday = _get_month_end(tday)
        new_df[cday] = df[tday]
    print(new_df.iloc[0:10])
    new_df.to_csv(os.path.join(path, 'industry_citic.csv'), encoding='gbk')
    '''