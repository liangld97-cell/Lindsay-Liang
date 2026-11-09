# -*- coding: utf-8 -*-
"""
==============================================================================
自建多策略 CTA 基准指数库构建系统 (cta_strategy_indices.py)
==============================================================================

【指数构建核心原则】
1. 固定 1.0 倍名义杠杆（自然波动率口径）：
   - 杜绝动态波动率缩放（No Vol-Timing），不因低波人为放大杠杆，也不因高波被动砍仓。
   - 真实反映各策略在市场顺风/逆风期的原始赚钱效应与周期弹性，作为公允的 FOF 比较标尺。
2. 趋势周期短/中/长三分法（基于实测平均持仓周期）：
   - 长周期趋势：持仓约 20 至 50 天（MA20/60 + 40日海龟突破 + 120日TSMOM）
   - 中周期趋势：持仓约 10 至 18 天（MA5/20  + 10日海龟突破 + 20日TSMOM）
   - 短周期趋势：持仓约 3 至 5 天（MA1/5   + 3日通道突破带中轨止盈 + 3日TSMOM）
3. 截面多空市值中性（周频调仓，降低无效换手）：
   - 截面强弱动量：做多 20/60日收益率前 20%（+50%），做空后 20%（-50%）
   - 期限结构Carry：做多展期收益率前 20% 贴水品种（+50%），做空后 20% 升水品种（-50%）
4. 复合CTA基准指数：
   - 25%长周期 + 20%中周期 + 10%短周期 + 25%截面动量 + 20%期限结构Carry

【输出文件】
- 仅导出单张标准的日度累计净值序列：`CTA策略基准指数_日度净值.csv`
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.universe import SYMBOLS
from cta_lib.portfolio import load_universe, make_panel, tradable_mask, estimate_cost_rates, portfolio_backtest
from cta_lib.data_loader import load_roll_yield_panel

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.unicode.east_asian_width', True)


# =============================================================================
# 一、权重与信号生成算子
# =============================================================================

def fixed_leverage_inv_vol_weights(sig, ret, vol_span=60, min_vol=0.05, target_gross_leverage=1.0):
    """
    固定名义杠杆定仓：品种间按过去 60 日波动率倒数逆波配权，组合总杠杆严格归一为 1.0 倍
    """
    vol = (ret.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(252)).clip(lower=min_vol)
    raw_w = (sig / vol).fillna(0.0)
    gross = raw_w.abs().sum(axis=1).replace(0.0, np.nan)
    return raw_w.div(gross, axis=0).mul(target_gross_leverage).fillna(0.0)


def _build_trend_long_signal(close, high, low):
    """
    长周期趋势信号（平均持仓约 20-50 个交易日）：
    - 双均线：MA(20, 60)
    - 通道突破：40 日唐奇安通道（双月高低点）
    - 时序动量：120 日动量（半年期收益方向）
    """
    ma_f, ma_s = close.rolling(20).mean(), close.rolling(60).mean()
    sig_ma = pd.DataFrame(np.where(ma_f > ma_s, 1.0, -1.0), index=close.index, columns=close.columns)

    up_40 = high.rolling(40).max().shift(1)
    dn_40 = low.rolling(40).min().shift(1)
    sig_don = pd.DataFrame(
        np.where(close > up_40, 1.0, np.where(close < dn_40, -1.0, np.nan)),
        index=close.index, columns=close.columns
    ).ffill().fillna(0.0)

    mom_120 = close / close.shift(120) - 1.0
    sig_ts = pd.DataFrame(np.where(mom_120 > 0, 1.0, -1.0), index=close.index, columns=close.columns)
    return (sig_ma + sig_don + sig_ts) / 3.0


def _build_trend_medium_signal(close, high, low):
    """
    中周期趋势信号（平均持仓约 10-18 个交易日）：
    - 双均线：MA(5, 20)
    - 通道突破：10 日唐奇安通道（双周高低点）
    - 时序动量：20 日动量（月度收益方向）
    """
    ma_f, ma_s = close.rolling(5).mean(), close.rolling(20).mean()
    sig_ma = pd.DataFrame(np.where(ma_f > ma_s, 1.0, -1.0), index=close.index, columns=close.columns)

    up_10 = high.rolling(10).max().shift(1)
    dn_10 = low.rolling(10).min().shift(1)
    sig_don = pd.DataFrame(
        np.where(close > up_10, 1.0, np.where(close < dn_10, -1.0, np.nan)),
        index=close.index, columns=close.columns
    ).ffill().fillna(0.0)

    mom_20 = close / close.shift(20) - 1.0
    sig_ts = pd.DataFrame(np.where(mom_20 > 0, 1.0, -1.0), index=close.index, columns=close.columns)
    return (sig_ma + sig_don + sig_ts) / 3.0


def _build_trend_short_signal(close, high, low):
    """
    短周期趋势信号（平均持仓约 3-5 个交易日，严格控制在一周内）：
    - 短均线：收盘价穿越 5 日均线（周均线）
    - 通道突破：3 日突破，并在回归 3 日通道中轨时迅速止盈平仓
    - 时序动量：3 日微动量
    """
    ma_5 = close.rolling(5).mean()
    sig_ma = pd.DataFrame(np.where(close > ma_5, 1.0, -1.0), index=close.index, columns=close.columns)

    up_3 = high.rolling(3).max().shift(1)
    dn_3 = low.rolling(3).min().shift(1)
    mid_3 = 0.5 * (up_3 + dn_3)

    sig_brk = pd.DataFrame(np.where(close > up_3, 1.0, np.where(close < dn_3, -1.0, 0.0)), index=close.index, columns=close.columns)
    in_upper = (close >= mid_3) & (close <= up_3) & (close.shift(1) > up_3.shift(1))
    in_lower = (close <= mid_3) & (close >= dn_3) & (close.shift(1) < dn_3.shift(1))
    sig_brk[in_upper] = 1.0
    sig_brk[in_lower] = -1.0

    mom_3 = close / close.shift(3) - 1.0
    sig_ts = pd.DataFrame(np.where(mom_3 > 0, 1.0, -1.0), index=close.index, columns=close.columns)
    return (sig_ma + sig_brk + sig_ts) / 3.0


def _weekly_rebalance_ls_weights(score_panel, tradable, top_q=0.20, rebalance_every=5):
    """
    截面多空定仓器（严格市值中性，固定 1.0 倍杠杆）：
    - 做多前 20%（总权重 +50%），做空后 20%（总权重 -50%），中间 60% 保持空仓
    - 每隔 5 个交易日调仓一次（周频换仓），大幅降低高频日度换手磨损
    """
    valid_score = score_panel.where(tradable & score_panel.notna())
    pct_rank = valid_score.rank(axis=1, ascending=False, pct=True)
    in_long = (pct_rank <= top_q)
    in_short = (pct_rank > (1.0 - top_q))

    w_long = in_long.div(in_long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_short = in_short.div(in_short.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_daily = 0.5 * w_long - 0.5 * w_short

    keep_mask = (np.arange(len(w_daily)) % rebalance_every == 0)
    w_weekly = w_daily.copy()
    w_weekly.loc[~keep_mask, :] = np.nan
    return w_weekly.ffill().fillna(0.0).where(tradable, 0.0)


# =============================================================================
# 二、全套基准指数回测与导出
# =============================================================================

def build_cta_style_indices(start_date='2020-01-01', end_date=None, save_csv=True, refresh=False, verbose=True):
    """
    一键构建七大 CTA 基准策略指数（日度收盘出信号，次日开盘 next_open 扣除滑点与主力换月成本）
    """
    if verbose:
        print(f"\n📊 正在加载数据并构建七大 CTA 策略基准指数 ({start_date} 至 最新)...")

    # 1. 基础行情与展期收益率加载
    data = load_universe(SYMBOLS, start=start_date, end=end_date, verbose=False)
    close = make_panel(data, 'close')
    high = make_panel(data, 'high')
    low = make_panel(data, 'low')
    ret = close.pct_change()
    tradable = tradable_mask(data).reindex(close.index).fillna(False)
    cost = estimate_cost_rates(data)

    roll_yield = load_roll_yield_panel(symbols=SYMBOLS, start=start_date, end=end_date, refresh=refresh)
    roll_yield = roll_yield.reindex(index=close.index, columns=close.columns).ffill()

    # 2. 生成各子策略目标权重
    sig_long = _build_trend_long_signal(close, high, low).where(tradable, 0.0)
    sig_med = _build_trend_medium_signal(close, high, low).where(tradable, 0.0)
    sig_short = _build_trend_short_signal(close, high, low).where(tradable, 0.0)
    sig_beta = pd.DataFrame(1.0, index=close.index, columns=close.columns).where(tradable, 0.0)

    combo_mom = 0.5 * (close / close.shift(20) - 1.0) + 0.5 * (close / close.shift(60) - 1.0)
    ry_smooth = roll_yield.rolling(5, min_periods=1).mean()

    weights_dict = {
        '长周期趋势': fixed_leverage_inv_vol_weights(sig_long, ret),
        '中周期趋势': fixed_leverage_inv_vol_weights(sig_med, ret),
        '短周期趋势': fixed_leverage_inv_vol_weights(sig_short, ret),
        '截面强弱动量': _weekly_rebalance_ls_weights(combo_mom, tradable),
        '期限结构Carry': _weekly_rebalance_ls_weights(ry_smooth, tradable),
        '商品多头Beta': fixed_leverage_inv_vol_weights(sig_beta, ret),
    }

    # 复合CTA基准（按资金权重 25%长趋势 + 20%中趋势 + 10%短趋势 + 25%截面动量 + 20%Carry 合成）
    weights_dict['复合CTA基准'] = (
        0.25 * weights_dict['长周期趋势'] +
        0.20 * weights_dict['中周期趋势'] +
        0.10 * weights_dict['短周期趋势'] +
        0.25 * weights_dict['截面强弱动量'] +
        0.20 * weights_dict['期限结构Carry']
    )

    # 3. 逐一扣费回测并归一化净值
    order_cols = ['复合CTA基准', '长周期趋势', '中周期趋势', '短周期趋势', '截面强弱动量', '期限结构Carry', '商品多头Beta']
    port_dict = {}
    for name in order_cols:
        port, _ = portfolio_backtest(data, weights_dict[name], cost_rate=cost, execution='next_open', roll_cost=True)
        port_dict[name] = port

    ret_df = pd.DataFrame({k: port_dict[k]['net_ret'] for k in order_cols})
    valid_start_idx = ret_df.index[65] if len(ret_df) > 65 else ret_df.index[0]  # 跳过 65 日预热期
    ret_df = ret_df.loc[valid_start_idx:].copy()
    nav_df = (1.0 + ret_df).cumprod()

    for k in order_cols:
        port_dict[k] = port_dict[k].loc[valid_start_idx:].copy()
        port_dict[k]['equity'] = nav_df[k]
        port_dict[k]['drawdown'] = nav_df[k] / nav_df[k].cummax() - 1.0

    # 4. 仅导出单张日度净值 CSV 文件
    if save_csv:
        csv_path = os.path.join(PROJECT_DIR, 'CTA策略基准指数_日度净值.csv')
        nav_df.to_csv(csv_path, encoding='utf-8-sig', float_format='%.6f')
        if verbose:
            print(f"✅ 日度净值已导出至: {csv_path}")

    return nav_df, ret_df, port_dict, None


# =============================================================================
# 三、每日监控看板打印
# =============================================================================

def print_daily_strategy_monitor(nav_df, ret_df, port_dict):
    """打印格式化每日监控看板（含实测持仓周期、近期涨跌幅、自然年化波动、夏普与最大回撤）"""
    last_dt = nav_df.index[-1]
    curr_year = str(last_dt.year)
    rows = []

    for col in nav_df.columns:
        eq = nav_df[col]
        dd = port_dict[col]['drawdown']
        r_1d = ret_df[col].iloc[-1]
        r_5d = eq.iloc[-1] / eq.iloc[-6] - 1.0 if len(eq) >= 6 else np.nan
        r_20d = eq.iloc[-1] / eq.iloc[-21] - 1.0 if len(eq) >= 21 else np.nan
        r_60d = eq.iloc[-1] / eq.iloc[-61] - 1.0 if len(eq) >= 61 else np.nan
        eq_prev_year = eq.loc[eq.index < f'{curr_year}-01-01']
        r_ytd = (eq.iloc[-1] / eq_prev_year.iloc[-1] - 1.0) if len(eq_prev_year) > 0 else (eq.iloc[-1] - 1.0)

        # 实测持仓周期：2 * 平均名义杠杆 / 日均双边换手率
        daily_turn = port_dict[col]['turnover'].mean()
        avg_lev = port_dict[col]['leverage'].mean()
        hold_days = (2.0 * avg_lev) / max(daily_turn, 1e-6) if col != '商品多头Beta' else np.inf

        ann_ret = (eq.iloc[-1]) ** (252.0 / len(eq)) - 1.0
        ann_vol = ret_df[col].std() * np.sqrt(252.0)
        sharpe = ann_ret / (ann_vol + 1e-12)

        rows.append({
            '持仓周期': '长期' if np.isinf(hold_days) else f'{hold_days:.1f}天',
            '今日(1D)': f"{r_1d*100:+6.2f}%",
            '本周(5D)': f"{r_5d*100:+6.2f}%",
            '本月(20D)': f"{r_20d*100:+6.2f}%",
            '近三月(60D)': f"{r_60d*100:+6.2f}%",
            '今年以来(YTD)': f"{r_ytd*100:+6.2f}%",
            '年化收益': f"{ann_ret*100:+5.2f}%",
            '最大回撤': f"{dd.min()*100:6.2f}%",
        })

    board = pd.DataFrame(rows, index=nav_df.columns)
    print("\n" + "═" * 105)
    print(f"📈 自建多策略 CTA 基准指数 · 每日收益监控看板 (截至 {last_dt.strftime('%Y-%m-%d')})")
    print("═" * 105)
    print(board.to_string())
    print("═" * 105 + "\n")
    return board


if __name__ == '__main__':
    nav_df, ret_df, port_dict, _ = build_cta_style_indices()
    print_daily_strategy_monitor(nav_df, ret_df, port_dict)
