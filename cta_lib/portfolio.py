# -*- coding: utf-8 -*-
"""
多品种组合回测

核心概念：
    weights（权重矩阵）：DataFrame，行 = 日期，列 = 品种代码
        weights.loc[t, 'RB'] = 0.05 表示：t 日收盘后，目标持有螺纹钢多头，名义市值 = 账户权益的 5%
        负数表示空头。所有品种 |权重| 之和 = 组合的名义杠杆（gross leverage）

    组合收益 = 各品种的（权重 × 收益 - 成本）之和
    执行延迟与单品种回测一样，由框架处理（默认 t 收盘出权重，t+1 开盘成交）
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_loader import load_dominant, init_rq, CACHE_DIR
from .backtest import run_backtest, setup_chinese_font
from .universe import UNIVERSE, sector_of, name_of


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def load_universe(symbols, start='2010-01-04', end=None, verbose=True):
    """批量加载主力连续合约日线，返回 {品种: df}。"""
    data = {}
    for i, s in enumerate(symbols, 1):
        data[s] = load_dominant(s, start=start, end=end, frequency='1d')
        if verbose:
            print(f'\r加载数据 {i}/{len(symbols)}: {s:<3}', end='')
    if verbose:
        print(' 完成')
    return data


def make_panel(data, field='close'):
    """把 {品种: df} 中的某一列拼成 日期 × 品种 的宽表。"""
    return pd.DataFrame({s: df[field] for s, df in data.items()}).sort_index()


def tradable_mask(data, min_volume=5000, vol_window=20, min_history=60):
    """
    可交易标记（True/False 宽表），只用当时及之前的信息：
        - 过去 vol_window 日平均成交量 >= min_volume 手（过滤流动性很差的时期，如早年的燃料油）
        - 上市后至少 min_history 个交易日（新品种上市初期价格行为不稳定）
    """
    vol = make_panel(data, 'volume')
    avg_vol = vol.rolling(vol_window, min_periods=vol_window).mean()
    history = vol.notna().cumsum()
    return (avg_vol >= min_volume) & (history >= min_history)


def estimate_cost_rates(data, commission=0.0001, slippage_ticks=1, refresh=False):
    """
    估算每个品种的单边交易成本（占名义市值比例）= 手续费 + N 跳滑点 / 最新价格
    例：玻璃 FG 价格约 800、最小变动 1 元 → 1 跳就是 12bp；铜 CU 价格 10 万、最小变动 10 元 → 1 跳只有 1bp
    tick_size 从米筐 instruments 获取并缓存。
    """
    path = os.path.join(CACHE_DIR, 'tick_sizes.json')
    ticks = {}
    if os.path.exists(path) and not refresh:
        with open(path, 'r', encoding='utf-8') as f:
            ticks = json.load(f)
    missing = [s for s in data if s not in ticks]
    if missing:
        rq = init_rq()
        for s in missing:
            ticks[s] = float(rq.instruments(data[s]['dominant_id'].iloc[-1]).tick_size())
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ticks, f)
    rates = {}
    for s, df in data.items():
        last_price = df['close'].iloc[-1]          # 前复权序列的最新价 = 真实最新价
        rates[s] = commission + slippage_ticks * ticks[s] / last_price
    return pd.Series(rates, name='单边成本')


# ----------------------------------------------------------------------
# 组合回测
# ----------------------------------------------------------------------
def portfolio_backtest(data, weights, cost_rate=0.0004, execution='next_open', roll_cost=True):
    """
    参数
    ----
    data      : {品种: df}
    weights   : 日期 × 品种 的目标权重宽表（NaN 视为 0）
    cost_rate : 统一费率（float），或每个品种不同的费率（dict / Series，如 estimate_cost_rates 的结果）

    返回
    ----
    port  : 组合层面 DataFrame —— gross_ret, cost, net_ret, equity, drawdown, leverage, turnover, n_assets
    asset : 各品种对组合的净收益贡献（日期 × 品种）
    """
    weights = weights.fillna(0.0)
    idx = weights.index
    net, gross, cost, pos = {}, {}, {}, {}
    for s in weights.columns:
        if s not in data:
            continue
        cr = cost_rate[s] if isinstance(cost_rate, (dict, pd.Series)) else cost_rate
        r = run_backtest(data[s], weights[s], cost_rate=cr,
                         execution=execution, roll_cost=roll_cost)
        net[s], gross[s], cost[s] = r['net_ret'], r['gross_ret'], r['cost']
        pos[s] = r['position']

    asset = pd.DataFrame(net).reindex(idx).fillna(0.0)
    port = pd.DataFrame(index=idx)
    port['gross_ret'] = pd.DataFrame(gross).reindex(idx).fillna(0.0).sum(axis=1)
    port['cost'] = pd.DataFrame(cost).reindex(idx).fillna(0.0).sum(axis=1)
    port['net_ret'] = asset.sum(axis=1)
    port['equity'] = (1 + port['net_ret']).cumprod()
    port['drawdown'] = port['equity'] / port['equity'].cummax() - 1
    pos_df = pd.DataFrame(pos).reindex(idx).fillna(0.0)
    port['leverage'] = pos_df.abs().sum(axis=1)
    port['turnover'] = pos_df.diff().abs().sum(axis=1)   # 每日权重变化量之和（多翻空算 2 倍）
    port['n_assets'] = (pos_df != 0).sum(axis=1)
    return port, asset


def vol_weights(sig, ret, vol_span=60, asset_vol_target=0.10, min_vol=0.05):
    """
    单品种波动率反比定仓（逆波动率/风险平价思想）：
    每个品种按其过去 vol_span 日的年化波动率倒数分配名义权重，
    波动大的品种仓位轻，波动小的品种仓位重。
    """
    single_vol = (ret.ewm(span=vol_span, min_periods=vol_span).std() * np.sqrt(252)).clip(lower=min_vol)
    n_active = (sig != 0).sum(axis=1).replace(0, np.nan)
    weights = (asset_vol_target / single_vol * sig).div(n_active, axis=0).fillna(0.0)
    return weights


def scale_to_target_vol(weights, returns, target_vol=0.10, lookback=60, rebalance_every=5,
                        max_leverage=None):
    """
    组合层面的波动率目标：根据“当前权重 × 过去 lookback 日的收益”估计组合波动，
    把整个组合按比例放大/缩小到 target_vol（年化）。只用历史信息。
    用途：让不同方案在【相同风险水平】下比较，回撤、收益才有可比性。

    rebalance_every：缩放系数每隔多少个交易日更新一次（默认 5 天 ≈ 每周）。
        如果每天都更新，缩放系数的微小波动会产生大量没有意义的小额调仓，白白多付成本。
    """
    w = weights.fillna(0.0)
    r = returns.reindex(index=w.index, columns=w.columns).fillna(0.0)
    # 用今天的权重乘以过去 lookback 日的收益，估计“如果过去一直持有今天这个组合”的波动
    rv, wv = r.values, w.values
    est = np.full(len(w), np.nan)
    for i in range(lookback - 1, len(w)):
        hist = rv[i - lookback + 1:i + 1]                   ## 过去60天45个品种的日收益率矩阵
        est[i] = (hist @ wv[i]).std() * np.sqrt(252)
    est_vol = pd.Series(est, index=w.index)                 ## 组合年化波动率
    scale = (target_vol / est_vol).replace([np.inf, -np.inf], np.nan)
    keep = np.arange(len(scale)) % rebalance_every == 0     ## 设定每隔 5个交易日（约每周）才重新测定一次系数
    scale = scale.where(keep).ffill().fillna(0.0)
    scaled = w.mul(scale, axis=0)                           # # 把45个品种的目标权重同比例乘上scale
    if max_leverage is not None:
        gross = scaled.abs().sum(axis=1)
        factor = (max_leverage / gross).clip(upper=1.0).fillna(1.0)
        scaled = scaled.mul(factor, axis=0)
    return scaled


# ----------------------------------------------------------------------
# 分析工具
# ----------------------------------------------------------------------
def sector_contribution(asset):
    """按板块汇总各品种的收益贡献（简单加总的累计贡献）。"""
    sectors = {s: sector_of(s) for s in asset.columns}
    by_sector = asset.T.groupby(sectors).sum().T
    return by_sector


def avg_pairwise_corr(asset_ret):
    """各品种策略收益之间的平均两两相关系数（只算两者都有交易的日子）。"""
    c = asset_ret.replace(0.0, np.nan).corr(min_periods=250)
    vals = c.values[np.triu_indices_from(c.values, k=1)]
    return np.nanmean(vals), c


def check_lookahead_panel(func, *panels, n_checks=4, **kwargs):
    """面板版未来函数检查：截断数据后重新计算，截断点之前的结果必须与全样本一致。"""
    full = func(*panels, **kwargs)
    n = len(full)
    for c in np.linspace(n * 0.4, n - 1, n_checks).astype(int):
        part = func(*[p.iloc[:c] for p in panels], **kwargs)
        a = full.iloc[:c].fillna(-999).values
        b = part.reindex(index=full.index[:c], columns=full.columns).fillna(-999).values
        if not np.allclose(a, b, atol=1e-10):
            print(f'[警告] 发现未来函数！截断到 {full.index[c - 1].date()} 时结果与全样本不一致')
            return False
    print('[通过] 未发现未来函数')
    return True


def plot_equity_compare(results, title='', log=True):
    """results: {名称: port DataFrame}，画净值与回撤对比。"""
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.3]})
    for name, port in results.items():
        axes[0].plot(port['equity'], label=name, lw=1.2)
        axes[1].plot(port['drawdown'], label=name, lw=1)
    if log:
        axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel('回撤')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bar(series, title='', color_by_sign=True):
    setup_chinese_font()
    s = series.sort_values()
    colors = ['tab:red' if v < 0 else 'tab:green' for v in s] if color_by_sign else None
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(s)), s.values, color=colors)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f'{name_of(k)}\n{k}' if k in UNIVERSE else k for k in s.index],
                       rotation=90, fontsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_corr(corr, title='品种策略收益相关性'):
    setup_chinese_font()
    order = sorted(corr.columns, key=lambda s: (sector_of(s), s))
    c = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(c.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels([f'{name_of(s)} {s}' for s in order], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
