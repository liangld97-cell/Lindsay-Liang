# -*- coding: utf-8 -*-
"""
期货数据加载（基于米筐 rqdatac）

核心函数：load_dominant(symbol, ...)
    获取某品种的“主力连续合约”行情，并缓存到本地 data_cache/，避免重复消耗流量。

关于连续合约的复权方式（重要）：
    主力合约换月时，新旧合约之间存在价差，直接拼接会产生“虚假跳空”。
    - adjust_type='none'                       不复权：有虚假跳空，不能直接算收益
    - adjust_method='prev_close_spread'        等差复权：消除跳空，但早期价格可能变负，pct_change 不准确
    - adjust_method='prev_close_ratio'（默认） 等比复权：收益率口径正确，做回测首选
"""
import os
import pandas as pd

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PROJECT_DIR, 'data_cache')
RQ_MIN_DATE = pd.Timestamp('2010-01-04')   # get_dominant_price 支持的最早日期

_RQ_INITED = False


def init_rq():
    """初始化 rqdatac（只初始化一次）。"""
    global _RQ_INITED
    import rqdatac
    if not _RQ_INITED:
        rqdatac.init()
        _RQ_INITED = True
    return rqdatac


def load_dominant(symbol, start=None, end=None, frequency='1d',
                  adjust_type='pre', adjust_method='prev_close_ratio',
                  refresh=False):
    """
    获取主力连续合约行情。

    参数
    ----
    symbol        : 品种代码，如 'RB'（螺纹钢）、'CU'（铜）、'I'（铁矿石）
    start, end    : 起止日期，字符串 'YYYY-MM-DD'。日线默认从 2010-01-04 开始；分钟线默认取最近 1 年
    frequency     : '1d' 日线，'1m' / '5m' / '15m' 等分钟线
    adjust_type   : 'pre' 前复权（默认），'post' 后复权，'none' 不复权
    adjust_method : 'prev_close_ratio' 等比（默认），'prev_close_spread' 等差
    refresh       : True 时忽略本地缓存，重新下载

    返回
    ----
    DataFrame，index 为时间，列包括：
        dominant_id   当天使用的主力合约代码
        open/high/low/close/volume/open_interest/settlement
        roll          是否为换月当天（True 表示从这一根 bar 开始使用新合约）
    """
    symbol = symbol.upper()
    if start is None:
        if frequency == '1d':
            start = RQ_MIN_DATE
        else:
            start = pd.Timestamp.today().normalize() - pd.DateOffset(years=1)
    start = max(pd.Timestamp(start), RQ_MIN_DATE)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()

    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = (f'{symbol}_{frequency}_{adjust_type}_{adjust_method}_'
             f'{start:%Y%m%d}_{end:%Y%m%d}.pkl')
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path) and not refresh:
        return pd.read_pickle(path)

    rq = init_rq()
    raw = rq.futures.get_dominant_price(
        symbol,
        start_date=start.strftime('%Y%m%d'),
        end_date=end.strftime('%Y%m%d'),
        frequency=frequency,
        adjust_type=adjust_type,
        adjust_method=adjust_method,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f'没有取到 {symbol} 的数据，请检查品种代码和日期范围')

    df = raw.droplevel(0).copy()          # 去掉 underlying_symbol 这一层索引
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date' if frequency == '1d' else 'datetime'

    cols = ['dominant_id', 'open', 'high', 'low', 'close',
            'volume', 'open_interest', 'settlement']
    df = df[[c for c in cols if c in df.columns]]

    prev_id = df['dominant_id'].shift(1)
    df['roll'] = df['dominant_id'].ne(prev_id) & prev_id.notna()

    df.to_pickle(path)
    return df


def load_roll_yield_panel(symbols=None, start=None, end=None, refresh=False):
    """
    加载全市场各品种历史日度展期收益率面板 (Roll Yield Panel).

    返回
    ----
    DataFrame (index: 日期 Timestamp, columns: underlying_symbol, values: annualized_yield_trading)
    """
    panel_path = os.path.join(CACHE_DIR, 'roll_yield_panel.pkl')
    raw_path = os.path.join(CACHE_DIR, 'roll_yield_20100104_20261023.pkl')

    piv = None
    if os.path.exists(panel_path) and not refresh:
        piv = pd.read_pickle(panel_path)
    elif os.path.exists(raw_path) and not refresh:
        raw = pd.read_pickle(raw_path)
        piv = raw.reset_index().pivot(index='date', columns='underlying_symbol', values='annualized_yield_trading')
        piv.index = pd.to_datetime(piv.index)
        piv.to_pickle(panel_path)

    # 检查是否需要自动增量拉取最新交易日的展期收益率
    req_end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    if piv is not None and not refresh:
        if piv.index.max() < req_end:
            try:
                rq = init_rq()
                from .universe import SYMBOLS
                delta_start = (piv.index.max() + pd.Timedelta(days=1)).strftime('%Y%m%d')
                delta_end = req_end.strftime('%Y%m%d')
                if delta_start <= delta_end:
                    delta_raw = rq.futures.get_roll_yield(SYMBOLS, start_date=delta_start, end_date=delta_end)
                    if delta_raw is not None and len(delta_raw) > 0:
                        delta_piv = delta_raw.reset_index().pivot(index='date', columns='underlying_symbol', values='annualized_yield_trading')
                        delta_piv.index = pd.to_datetime(delta_piv.index)
                        piv = pd.concat([piv[~piv.index.isin(delta_piv.index)], delta_piv]).sort_index()
                        piv.to_pickle(panel_path)
            except Exception as e:
                pass
    elif piv is None:
        rq = init_rq()
        from .universe import SYMBOLS
        syms = symbols if symbols is not None else SYMBOLS
        s_str = pd.Timestamp(start).strftime('%Y%m%d') if start else '20100104'
        e_str = pd.Timestamp(end).strftime('%Y%m%d') if end else pd.Timestamp.today().strftime('%Y%m%d')
        raw = rq.futures.get_roll_yield(syms, start_date=s_str, end_date=e_str)
        piv = raw.reset_index().pivot(index='date', columns='underlying_symbol', values='annualized_yield_trading')
        piv.index = pd.to_datetime(piv.index)
        piv.to_pickle(panel_path)

    if symbols is not None:
        valid_cols = [s for s in symbols if s in piv.columns]
        piv = piv[valid_cols]
    if start is not None:
        piv = piv.loc[piv.index >= pd.Timestamp(start)]
    if end is not None:
        piv = piv.loc[piv.index <= pd.Timestamp(end)]
    return piv


def load_warehouse_panel(symbols=None, start=None, end=None, refresh=False):
    """
    加载全市场各品种交易所注册仓单量面板 (Warehouse Stocks / On-Warrant Panel).

    返回
    ----
    DataFrame (index: 日期 Timestamp, columns: underlying_symbol, values: on_warrant 注册仓单量)
    """
    panel_path = os.path.join(CACHE_DIR, 'warehouse_panel.pkl')
    raw_path = os.path.join(CACHE_DIR, 'warehouse_20100104_20261023.pkl')

    piv = None
    if os.path.exists(panel_path) and not refresh:
        piv = pd.read_pickle(panel_path)
    elif os.path.exists(raw_path) and not refresh:
        raw = pd.read_pickle(raw_path)
        piv = raw['on_warrant'].unstack('underlying_symbol')
        piv.index = pd.to_datetime(piv.index.astype(str))
        piv = piv.astype(float).sort_index()
        piv.to_pickle(panel_path)

    # 检查是否需要自动增量拉取最新交易日的注册仓单数据
    req_end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    if piv is not None and not refresh:
        if piv.index.max() < req_end:
            try:
                rq = init_rq()
                from .universe import SYMBOLS
                delta_start = (piv.index.max() + pd.Timedelta(days=1)).strftime('%Y%m%d')
                delta_end = req_end.strftime('%Y%m%d')
                if delta_start <= delta_end:
                    delta_raw = rq.futures.get_warehouse_stocks(SYMBOLS, start_date=delta_start, end_date=delta_end)
                    if delta_raw is not None and len(delta_raw) > 0:
                        delta_piv = delta_raw['on_warrant'].unstack('underlying_symbol')
                        delta_piv.index = pd.to_datetime(delta_piv.index.astype(str))
                        delta_piv = delta_piv.astype(float).sort_index()
                        piv = pd.concat([piv[~piv.index.isin(delta_piv.index)], delta_piv]).sort_index()
                        piv.to_pickle(panel_path)
            except Exception:
                pass
    elif piv is None:
        rq = init_rq()
        from .universe import SYMBOLS
        syms = symbols if symbols is not None else SYMBOLS
        s_str = pd.Timestamp(start).strftime('%Y%m%d') if start else '20100104'
        e_str = pd.Timestamp(end).strftime('%Y%m%d') if end else pd.Timestamp.today().strftime('%Y%m%d')
        raw = rq.futures.get_warehouse_stocks(syms, start_date=s_str, end_date=e_str)
        piv = raw['on_warrant'].unstack('underlying_symbol')
        piv.index = pd.to_datetime(piv.index.astype(str))
        piv = piv.astype(float).sort_index()
        piv.to_pickle(panel_path)

    if symbols is not None:
        valid_cols = [s for s in symbols if s in piv.columns]
        piv = piv[valid_cols]
    if start is not None:
        piv = piv.loc[piv.index >= pd.Timestamp(start)]
    if end is not None:
        piv = piv.loc[piv.index <= pd.Timestamp(end)]
    return piv


def load_member_rank_panel(symbols=None, start=None, end=None, refresh=False):
    """
    加载全市场各品种前 20 名期货公司会员持仓量汇总面板 (Member Rank Panel).

    返回
    ----
    dict:
        'long'  : DataFrame (日期 × 品种，前 20 名会员多头持仓量合计)
        'short' : DataFrame (日期 × 品种，前 20 名会员空头持仓量合计)
    """
    raw_path = os.path.join(CACHE_DIR, 'member_rank_20150105_20261023.pkl')
    if os.path.exists(raw_path) and not refresh:
        out = pd.read_pickle(raw_path)
    else:
        raise FileNotFoundError(f"未找到本地会员持仓缓存文件: {raw_path}")

    # 检查是否需要自动增量拉取最新交易日的会员持仓数据
    req_end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    max_dt = pd.to_datetime(out['long'].index).max()
    if max_dt < req_end and not refresh:
        try:
            rq = init_rq()
            from .universe import SYMBOLS
            delta_start = (max_dt + pd.Timedelta(days=1)).strftime('%Y%m%d')
            delta_end = req_end.strftime('%Y%m%d')
            if delta_start <= delta_end:
                updated = False
                for side in ['long', 'short']:
                    cols = {}
                    for s in SYMBOLS:
                        try:
                            df = rq.futures.get_member_rank(s, start_date=delta_start, end_date=delta_end, rank_by=side)
                            if df is not None and len(df) > 0:
                                cols[s] = df.groupby('trading_date')['volume'].sum()
                        except Exception:
                            pass
                    if cols:
                        delta_df = pd.DataFrame(cols)
                        delta_df.index = pd.to_datetime(delta_df.index)
                        old_df = out[side].copy()
                        old_df.index = pd.to_datetime(old_df.index)
                        out[side] = pd.concat([old_df[~old_df.index.isin(delta_df.index)], delta_df]).sort_index()
                        updated = True
                if updated:
                    pd.to_pickle(out, raw_path)
        except Exception:
            pass

    res = {}
    for side, df in out.items():
        sub = df.copy()
        sub.index = pd.to_datetime(sub.index)
        sub = sub.sort_index()
        if symbols is not None:
            valid_cols = [s for s in symbols if s in sub.columns]
            sub = sub[valid_cols]
        if start is not None:
            sub = sub.loc[sub.index >= pd.Timestamp(start)]
        if end is not None:
            sub = sub.loc[sub.index <= pd.Timestamp(end)]
        res[side] = sub
    return res


