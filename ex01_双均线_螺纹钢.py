# -*- coding: utf-8 -*-
"""
练习 01：螺纹钢双均线策略
============================================================
用 Spyder 打开本文件，按 Ctrl+Enter 逐个运行下面的 #%% 代码块。

你的任务：
    只需要完成【第 2 块】里的 generate_signal() 函数。
    其余代码（数据、回测、作图、指标）都已经写好。

规则：
    快线 MA(fast) 在慢线 MA(slow) 之上 → 做多（+1）
    快线 MA(fast) 在慢线 MA(slow) 之下 → 做空（-1）
    均线还没有足够数据计算时（序列开头）→ 空仓（0）

写完之后：
    1. 运行第 3 块“未来函数检查”，必须显示 [通过]
    2. 运行第 4 块看回测结果
    3. 回答文件末尾的思考题（可以直接写在注释里，发给我一起讨论）
"""

#%% 第 0 块：环境准备 ----------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd

try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:   # 某些 Spyder 运行方式下没有 __file__
    PROJECT_DIR = r'c:\Users\Administrator\Desktop\FOF材料\CTA学习\CTA学习'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.data_loader import load_dominant
from cta_lib.backtest import (run_backtest, performance, yearly_returns,
                              plot_result, check_lookahead, trade_list)

pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')


#%% 第 1 块：取数据（第一次会从米筐下载，之后读本地缓存）----------------
df = load_dominant('RB', start='2010-01-04', end='2026-10-23', frequency='1d')


# 小练习：在变量浏览器里打开 df，找一行 roll == True 的数据，
#         看看换月前后 dominant_id 是怎么变化的；回答：有些月份会更换主力合约，有些月份不会


#%% 第 2 块：★ 你来写 ★ ------------------------------------------------
def generate_signal(df, fast=20, slow=60):
    """
    输入
        df   : 行情 DataFrame，可以用 df['close'] 等列
        fast : 快线窗口
        slow : 慢线窗口
    输出
        pd.Series，index 和 df.index 完全一致，取值为 1 / -1 / 0

    提示
        - pandas 的 .rolling(n).mean() 可以计算均线
        - np.where(条件, 值1, 值2) 或者 pd.Series 的布尔运算都可以
        - 开头均线为 NaN 的部分要设为 0
        - 不要自己 shift！执行延迟由回测框架处理（详见 backtest.py 开头的说明）
    """
    # ======== 在这里写你的代码 ========
    f = df['close'].rolling(fast).mean()
    s = df['close'].rolling(slow).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[f > s] = 1.0
    # 空头那一行你来补
    signal[f < s] = -1
    # ==================================
    return signal


#%% 第 3 块：未来函数检查 ----------------------------------------------
sig = generate_signal(df, fast=60, slow=250)
print(sig.value_counts())
check_lookahead(generate_signal, df, fast=20, slow=60)


#%% 第 4 块：回测与结果 ------------------------------------------------
res = run_backtest(df, sig, cost_rate=0.0004, execution='next_open')

print(performance(res).to_string())
print('\n分年度收益：')
print(yearly_returns(res, df))
# plot_result(res, df, title='螺纹钢 双均线(20, 60)')


#%% 第 5 块：看看每笔交易 ----------------------------------------------
trades = trade_list(res)
print(trades.sort_values('收益').head(5))    # 亏得最多的 5 笔
print(trades.sort_values('收益').tail(5))    # 赚得最多的 5 笔
# 观察：赚钱的交易和亏钱的交易，持有时间有什么区别？


#%% 第 6 块：对照实验（第 4 块跑通后再做）----------------------------
# (a) 理想化成交 vs 次日开盘成交
res_close = run_backtest(df, sig, cost_rate=0.0004, execution='close')
# (b) 不计成本 vs 成本翻倍
res_free = run_backtest(df, sig, cost_rate=0.0)
res_2x = run_backtest(df, sig, cost_rate=0.0008)

compare = pd.DataFrame({
    '基准(次日开盘,4bp)': performance(res),
    '收盘成交': performance(res_close),
    '零成本': performance(res_free),
    '成本翻倍': performance(res_2x),
})
print(compare.loc[['年化收益', '夏普比率', '最大回撤', '交易次数']])
print(res['cost'].sum() / res['gross_ret'].sum())   # 成本吃掉了毛收益的多少
print(res['cost'].sum() / len(res) * 252)           # 年化成本

#%% 思考题（写在下面的注释里）------------------------------------------
# Q1. 这个策略的交易胜率大概是多少？盈亏比呢？这对“趋势策略”意味着什么？
# 回答：交易的月度胜率是50.99%，盈亏比是2.1667，胜率低但赔率高

# Q2. 日收益偏度是正还是负？结合 Q1 解释原因。
#     如果某个管理人自称做“趋势跟踪”，但净值的偏度明显为负，你会怀疑什么？
#回答：日度胜率是正偏，但是只是稍微正偏；正偏的意思是出现正收益的情况要多过于负收益的情况，相对于正态分布来说；如果
# 如果某个管理人自称做“趋势跟踪”，但净值的偏度明显为负，但我会觉得他捕捉收益的能力不行，比如在某些品种出现行情之前就平仓了，没有吃到收益

# Q3. 分年度来看，哪几年赚得最多、哪几年亏损？回忆一下那几年螺纹钢的行情。
#  2010-2016都是正收益，其中2016年的收益有108%，这几年这个策略的变现都显著好于买入持有；2020-2022；2025-2026也都是正收益
# 2017-2019、2023-2024都是亏钱的

# Q4. 第 6 块里“收盘成交”比“次日开盘”好多少？我不理解这个问题，看指标的话也就好一点吧
#     尽调时，管理人给你看的回测是用哪种成交假设，为什么值得问？
#
# Q5. 把参数改成 (5, 20) 和 (60, 250) 各跑一次，交易次数、成本占比、
#     夏普分别怎么变化？短周期策略对什么因素最敏感？

## （5，20）交易次数235次，夏普是0.153
##  （20，60）交易次数79次，夏普是0.1282
##  （60，250）交易次数6次，成本占比是？，夏普是0.2936




