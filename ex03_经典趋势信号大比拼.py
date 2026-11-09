# -*- coding: utf-8 -*-
"""
练习 03：四大经典时序趋势信号大比拼与多信号集成
============================================================
用 Spyder 打开，按 Ctrl+Enter 逐块运行。

【背景与目标】
在 FOF 尽调中，经常听到管理人说：
    “我们的底层策略融合了多种不同的趋势信号（均线、通道突破、时序动量），
     做到了多周期、多信号的低相关性互补。”
这是真的还是噱头？

本练习我们将在 45 个商品期货品种的大样本池上，亲自把这四大经典信号全部跑一遍：
    1. 双均线系统（Dual Moving Average）—— 练习 01/02 已掌握，作为基准
    2. 唐奇安通道突破（Donchian / 海龟交易法）—— ★你来写★
    3. 布林带通道突破（Bollinger Bands Breakout）—— ★你来写★
    4. 时间序列动量（TSMOM / 跨周期动量）—— ★你来写★

最后，我们将四大信号进行“集成对冲”，看看为什么管理人都要做多信号混合！
"""

#%% 第 0 块：环境准备 ----------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_DIR = r'C:\Users\Lenovo\Desktop\CTA学习'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.universe import SYMBOLS, name_of, sector_of
from cta_lib.backtest import performance, yearly_returns, check_lookahead, setup_chinese_font
from cta_lib.portfolio import (load_universe, make_panel, tradable_mask, estimate_cost_rates,
                               vol_weights, scale_to_target_vol, portfolio_backtest,
                               plot_equity_compare, plot_corr)

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

START, END = '2010-01-04', '2026-10-30'
TARGET_VOL = 0.10            # 统一将所有策略缩放到 10% 目标波动率，公平对比


#%% 第 1 块：数据加载与预处理 ------------------------------------------
data = load_universe(SYMBOLS, start=START, end=END)
close = make_panel(data, 'close')
ret = close.pct_change()
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)


#%% 第 2 块：基准信号 1 —— 双均线（MA）---------------------------------
def sig_ma(df, fast=20, slow=60):
    f = df['close'].rolling(fast).mean()
    s = df['close'].rolling(slow).mean()
    sig = pd.Series(0.0, index=df.index)
    sig[f > s] = 1.0
    sig[f < s] = -1.0
    return sig


#%% 第 3 块： 信号 2 —— 唐奇安通道（海龟交易法则）---------
def sig_donchian(df, window=20):
    """
    【唐奇安通道突破（海龟交易法核心）】
    规则：
        1. 当今天收盘价 > 过去 window 日的最高价（不含今天！）时，突破做多（+1.0）
        2. 当今天收盘价 < 过去 window 日的最低价（不含今天！）时，跌破做空（-1.0）
        3. 突破发生后，持续持仓，直到发生反向突破（海龟持仓状态：用 .ffill() 顺延）
        
    特点：如果某天行情出现一根剧烈的“长上下影线插针”，通道会被瞬间撑开，而且需要等待整整 20 天才能把这个极端值移出窗口。
    """
   
    f = df['high'].rolling(window).max().shift(1) 
    s = df['low'].rolling(window).min().shift(1)
    sig = pd.Series(np.nan, index=df.index)
    sig[df['close'] > f] = 1.0
    sig[df['close'] < s] = -1.0
    sig=sig.ffill().fillna(0)
   
    return sig


#%% 第 4 块：信号 3 —— 布林带通道突破 --------------------
def sig_bollinger(df, window=20, n_std=1.5):
    """
    【布林带突破策略（自适应波动率通道）】
    规则：
        1. 中轨（Mid）：过去 window 日收盘价的均线
        2. 轨道宽度：过去 window 日收盘价的标准差 std
        3. 上轨（Upper）：Mid + n_std * std（昨日上轨，需 shift(1)）
        4. 下轨（Lower）：Mid - n_std * std（昨日下轨，需 shift(1)）
        5. 当收盘价突破昨日上轨时做多（+1.0），跌破昨日下轨时做空（-1.0），用 .ffill() 保持持仓
        
    特点：对极值具有“统计平滑和波动自适应”能力（软边界）
    """

    mid=df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    upper = (mid + n_std * std).shift(1)
    lower =  (mid - n_std * std).shift(1)
    sig=pd.Series(np.nan,index=df.index)
    sig[df['close']>upper] = 1
    sig[df['close']<lower] = -1
    sig=sig.ffill().fillna(0)

    return sig


#%% 第 5 块：信号 4 —— 时间序列动量（TSMOM）--------------
def sig_tsmom(df, lookback=250):
    """
    【时间序列动量（Moskowitz, Ooi, Pedersen 2012 经典论文）】
    规则：
        如果过去 lookback 日（通常为 12 个月/约 250 个交易日）的累计收益率为正，做多（+1.0）；
        如果过去 lookback 日的累计收益率为负，做空（-1.0）；
        开头不足 lookback 日时保持空仓（0.0）。
    """
    f = df['close'] / df['close'].shift(lookback) -1
    sig=pd.Series(np.nan,index=df.index)
    sig[f > 0] = 1.0  
    sig[f < 0] = -1.0  
    
    return sig


#%% 第 6 块：四大信号的未来函数体检 -----------------------------------
print('\n===== 四大信号未来函数检验 =====')
for name, fn, kw in [('双均线', sig_ma, {'fast': 20, 'slow': 60}),
                     ('唐奇安海龟', sig_donchian, {'window': 20}),
                     ('布林带突破', sig_bollinger, {'window': 20, 'n_std': 1.5}),
                     ('TSMOM动量', sig_tsmom, {'lookback': 250})]:
    print(f'正在体检: {name}...')
    check_lookahead(fn, data['RB'], **kw)


#%% 第 7 块：全市场 45 个品种回测与【四信号集成】-----------------------
# 1. 批量计算四大信号的面板宽表
sig_dict = {}
for name, fn, kw in [('双均线', sig_ma, {'fast': 20, 'slow': 60}),
                     ('唐奇安海龟', sig_donchian, {'window': 20}),
                     ('布林带突破', sig_bollinger, {'window': 20, 'n_std': 1.5}),
                     ('TSMOM动量', sig_tsmom, {'lookback': 250})]:
    p_sig = pd.DataFrame({s: fn(data[s], **kw) for s in SYMBOLS})
    sig_dict[name] = p_sig.reindex(close.index).fillna(0.0).where(tradable, 0.0)

# 2. 构建【多信号集成（Ensemble）】：将四大信号等权平均！
#    仓位不再只是 +1 或 -1，可能出现 +0.75, +0.5, +0.25, 0 等平滑过渡
sig_dict['四信号集成'] = (sig_dict['双均线'] + sig_dict['唐奇安海龟'] +
                        sig_dict['布林带突破'] + sig_dict['TSMOM动量']) / 4.0

# 3. 运行多品种回测（全部应用练习 02 的波动率平价定仓与 10% 目标波动）
results = {}
for name, p_sig in sig_dict.items():
    w = vol_weights(p_sig, ret)
    w_scaled = scale_to_target_vol(w, ret, target_vol=TARGET_VOL)
    port, _ = portfolio_backtest(data, w_scaled, cost_rate=cost)
    results[name] = port

perf_df = pd.DataFrame({name: performance(port) for name, port in results.items()})
print('\n===== 四大信号与集成组合 绩效总览（均为 10% 目标波动率）=====')
print(perf_df.loc[['年化收益', '年化波动', '夏普比率', '索提诺比率', '最大回撤', '卡玛比率',
                    '日收益偏度', '月收益偏度', '月度胜率', '年化换手(倍)', '年化成本']])

plot_equity_compare(results, title='四大经典趋势信号 vs 四信号集成（10% 目标波动率）')


#%% 第 8 块：信号间相关性矩阵（为什么能互补？）-----------------------
rets = pd.DataFrame({name: port['net_ret'] for name, port in results.items()})
corr_matrix = rets[['双均线', '唐奇安海龟', '布林带突破', 'TSMOM动量']].corr()
print('\n===== 四大信号净收益相关系数矩阵 =====')
print(corr_matrix.round(3))
plot_corr(corr_matrix, title='四大趋势策略收益相关性')


#%% 第 9 块：分年度表现对比 --------------------------------------------
yr = pd.DataFrame({name: yearly_returns(port)['策略'] for name, port in results.items()})
print('\n===== 分年度收益对比 =====')
print(yr)


#%% 思考题（写在下面的注释里）------------------------------------------
# Q1.【信号对比】双均线、海龟、布林带、TSMOM 中，哪一个单信号的夏普最高？
#     海龟和布林带的“月收益偏度”表现如何？谁抓肥尾的能力最强？
#月度收益偏度都是正偏，抓肥尾的能力不应该是看峰度这个指标么

# Q2.【相关性之谜】观察第 8 块的相关系数矩阵：
#     (a) 哪两个信号之间的相关性最高（甚至接近 0.9）？为什么它们的逻辑这么像？ 海龟交易和布林带突破相关性很高，
#     (b) 哪一个信号和其他三个信号的相关性最低（< 0.5）？为什么？  TSMOM动量和其他三个的相关性最低，可能是因为他的时间区间比较长？

#
# Q3.【集成效果】对比“四信号集成”和单个信号的表现：
#     集成的月度胜率、最大回撤、夏普比率有什么变化？ 会相较于某些单一的策略胜率变高，回撤变小，但也会相较于某些策略反过来
#     为什么把几个夏普不一的信号混在一起，反而能让净值更稳健？ 因为分散了
#
# Q4.【震荡市表现】在商品市场反复拉锯震荡的年份（如 2017、2019、2023）：
#     单一均线策略亏了多少？四信号集成的回撤控制得怎么样？
#
# Q5.【尽调拷问】一个 CTA 管理人宣称：“我们模型融合了 20 个不同的技术指标趋势信号”。
#     结合今天的相关性实验，如果你去尽调他，你会怎么当面验证他这 20 个信号是“真正互补”，
#     还是“换汤不换药的同质化信号”？ 看策略相关性和组合的收益夏普这类的？
#
# Q6.【换手与手续费】四大信号中，哪一个信号年化换手最高、手续费最重？哪一个最省成本？ 布林带突破换手率最高手续费最贵；双均线策略最省成本
#     对于管理规模超过 20 亿的大型 CTA，你更倾向于配置哪种风格的趋势信号？ 双均线
