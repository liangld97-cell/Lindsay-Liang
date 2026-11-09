# -*- coding: utf-8 -*-
"""
练习 02：从单品种到多品种组合
============================================================
用 Spyder 打开，按 Ctrl+Enter 逐块运行。

本练习要回答练习 01 留下的问题：“双均线这个信号到底靠不靠谱？”

    A 部分  广度检验：同一个信号、同一组参数，在 45 个品种上各跑一遍
    B 部分  等权组合：把所有品种放在一起，看分散化能带来什么
    C 部分  ★你来写★ 波动率调仓：让每个品种贡献差不多的风险
    D 部分  最终组合的分年度表现

为了公平比较，B、C 部分的组合最后都会统一缩放到 年化 10% 的目标波动率
（scale_to_target_vol），这样比较回撤和收益才有意义。
"""

#%% 第 0 块：环境准备 ----------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd

try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_DIR = r'c:\Users\Administrator\Desktop\FOF材料\CTA学习\CTA学习'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.backtest import run_backtest, performance, yearly_returns
from cta_lib.universe import SYMBOLS, name_of, sector_of
from cta_lib.portfolio import (load_universe, make_panel, tradable_mask, estimate_cost_rates,
                               portfolio_backtest, scale_to_target_vol, sector_contribution,
                               avg_pairwise_corr, check_lookahead_panel,
                               plot_equity_compare, plot_bar, plot_corr)

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

START, END = '2010-01-04', '2026-10-30'
FAST, SLOW = 20, 60          # 均线参数统一在这里改
TARGET_VOL = 0.10            # 组合目标年化波动率


#%% 第 1 块：数据、可交易标记、交易成本 --------------------------------
data = load_universe(SYMBOLS, start=START, end=END)

close = make_panel(data, 'close')                 # 日期 × 品种 收盘价（前复权）
ret = close.pct_change()                          # 日收益率
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)                  # 每个品种的单边成本（手续费 + 1 跳滑点）

print('\n单边交易成本最高的 5 个品种：')
print(cost.sort_values().tail(5).rename(index=lambda s: f'{name_of(s)} {s}'))
print('\n单边交易成本最低的 5 个品种：')
print(cost.sort_values().head(5).rename(index=lambda s: f'{name_of(s)} {s}'))



#%% 第 2 块：信号函数（就是你在练习 01 里写的，原样搬过来）-------------
def generate_signal(df, fast=5, slow=20):
    f = df['close'].rolling(fast).mean()
    s = df['close'].rolling(slow).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[f > s] = 1.0
    signal[f < s] = -1.0
    return signal


# 所有品种的信号拼成宽表；不可交易的日子信号置 0
sig = pd.DataFrame({s: generate_signal(data[s], FAST, SLOW) for s in SYMBOLS})
sig = sig.reindex(close.index).fillna(0.0).where(tradable, 0.0)
print(f'信号宽表：{sig.shape[0]} 天 × {sig.shape[1]} 个品种')


#%% ===================== A 部分：广度检验 =============================
rows = {}
for s in SYMBOLS:
    res_s = run_backtest(data[s], sig[s], cost_rate=cost[s])
    res_s = res_s[sig[s].reindex(res_s.index).ne(0).cummax()]   # 从第一次有信号开始算
    p = performance(res_s)
    rows[s] = {'名称': name_of(s), '板块': sector_of(s),
               '年数': len(res_s) / 252, '夏普': p['夏普比率'],
               '年化收益': p['年化收益'], '最大回撤': p['最大回撤'],
               '交易胜率': p['交易胜率'], '盈亏比': p['盈亏比']}
breadth = pd.DataFrame(rows).T
for c in ['年数', '夏普', '年化收益', '最大回撤', '交易胜率', '盈亏比']:
    breadth[c] = breadth[c].astype(float)

print(breadth.sort_values('夏普', ascending=False).to_string())
print(f'\n夏普 > 0 的品种：{(breadth["夏普"] > 0).sum()} / {len(breadth)}')
print(f'单品种夏普 平均值：{breadth["夏普"].mean():.3f}，中位数：{breadth["夏普"].median():.3f}')
print('\n各板块平均夏普：')
print(breadth.groupby('板块')['夏普'].agg(['mean', 'count']))

plot_bar(breadth['夏普'], title=f'双均线({FAST},{SLOW}) 各品种夏普比率')


#%% ===================== B 部分：等权组合 =============================
# 每天把资金平均分给所有“可交易”的品种：权重 = 信号 / 当天可交易品种数
n_tradable = tradable.sum(axis=1).replace(0, np.nan) 
W_eq = sig.div(n_tradable, axis=0).fillna(0.0) 
port_eq_raw, asset_eq_raw = portfolio_backtest(data, W_eq, cost_rate=cost)

# 组合当天的预估年化等于过去60天组合的日收益率序列的年化波动率，每五天调仓一次。
W_eq10 = scale_to_target_vol(W_eq, ret, target_vol=TARGET_VOL) 
port_eq, asset_eq = portfolio_backtest(data, W_eq10, cost_rate=cost)

# 对照组：只做螺纹钢，同样缩放到 10% 波动
W_rb10 = scale_to_target_vol(sig[['RB']], ret[['RB']], target_vol=TARGET_VOL)
port_rb, _ = portfolio_backtest(data, W_rb10, cost_rate=cost)

compare_B = pd.DataFrame({
    '螺纹钢单品种(10%波动)': performance(port_rb),
    '等权组合(1倍名义)': performance(port_eq_raw),
    '等权组合(10%波动)': performance(port_eq),
})
print(compare_B.to_string())
plot_equity_compare({'螺纹钢单品种': port_rb, '等权组合': port_eq},
                    title=f'单品种 vs 多品种（均为 {TARGET_VOL:.0%} 目标波动）')


#%% 第 B2 块：为什么分散有用？——相关性 -------------------------------
rho, corr = avg_pairwise_corr(asset_eq)
N = len(SYMBOLS)
sr_avg = breadth['夏普'].mean()
sr_theory = sr_avg * np.sqrt(N / (1 + (N - 1) * rho))
print(f'各品种策略收益的平均两两相关系数 ρ = {rho:.3f}')
print(f'理论组合夏普 ≈ 平均单品种夏普 × sqrt(N / (1 + (N-1)ρ)) = '
      f'{sr_avg:.3f} × {np.sqrt(N / (1 + (N - 1) * rho)):.2f} = {sr_theory:.3f}')
print(f'实际组合夏普 = {performance(port_eq)["夏普比率"]:.3f}')
plot_corr(corr)
# 观察热力图：哪些品种之间相关性高？是不是同一板块的？


#%% 第 B3 块：各板块贡献 ----------------------------------------------
sec = sector_contribution(asset_eq)
print('各板块累计收益贡献（简单加总）：')
print(sec.sum().sort_values(ascending=False))
plot_equity_compare({k: pd.DataFrame({'equity': 1 + sec[k].cumsum(),
                                      'drawdown': (1 + sec[k].cumsum()) / (1 + sec[k].cumsum()).cummax() - 1})
                     for k in sec.columns}, title='各板块累计贡献', log=False)


#%% ===================== C 部分：波动率调仓 ================
def vol_weights(sig, ret, vol_span=60, asset_vol_target=0.10, min_vol=0.05):
    
    single_vol = (ret.ewm(span=vol_span,min_periods=vol_span).std()*np.sqrt(252)).clip(lower=min_vol)
    n_active = (sig!=0).sum(axis=1).replace(0,np.nan)
    weights = (asset_vol_target/single_vol * sig).div(n_active,axis=0).fillna(0.0) 

    return weights


#%% 第 C2 块：未来函数检查 + 回测 -------------------------------------
W_vol = vol_weights(sig, ret)
check_lookahead_panel(vol_weights, sig, ret)

W_vol10 = scale_to_target_vol(W_vol, ret, target_vol=TARGET_VOL)
port_vol, asset_vol = portfolio_backtest(data, W_vol10, cost_rate=cost)

compare_C = pd.DataFrame({
    '螺纹钢单品种': performance(port_rb),
    '等权组合': performance(port_eq),
    '波动率调仓组合': performance(port_vol)})
print(compare_C.to_string())
plot_equity_compare({'螺纹钢单品种': port_rb, '等权组合': port_eq, '波动率调仓': port_vol},
                    title=f'三种方案对比（均为 {TARGET_VOL:.0%} 目标波动）')


#%% 第 C3 块：风险贡献对比 --------------------------------------------
# 用“各品种对组合日收益贡献的标准差占比”近似衡量风险贡献
def risk_share(asset):
    s = asset.std()
    return s / s.sum()

risk_cmp = pd.DataFrame({'等权': risk_share(asset_eq), '波动率调仓': risk_share(asset_vol)})
risk_cmp.index = [f'{name_of(s)} {s}' for s in risk_cmp.index]
print('等权组合中风险贡献最大的 8 个品种：')
print(risk_cmp.sort_values('等权', ascending=False).head(8))
print('\n按板块汇总的风险占比：')
risk_sec = pd.DataFrame({'等权': risk_share(asset_eq), '波动率调仓': risk_share(asset_vol)})
print(risk_sec.groupby(lambda s: sector_of(s)).sum())


#%% ===================== D 部分：分年度表现 ============================
yr = pd.DataFrame({
    '螺纹钢单品种': yearly_returns(port_rb)['策略'],
    '等权组合': yearly_returns(port_eq)['策略'],
    '波动率调仓': yearly_returns(port_vol)['策略'],
})
print(yr)


#%% 思考题（写在下面的注释里）------------------------------------------
#
# Q3.【风险贡献】等权组合里，风险贡献最大的是哪几个品种 / 哪个板块？
#     波动率调仓之后有什么变化？夏普和最大回撤有改善吗？
#
# Q4.【产品特征】最终组合的日收益偏度、月收益偏度是正还是负？和练习 01 单品种比有什么变化？
#     最长回撤期有多久（换算成月）？如果你是 FOF 投资经理，能接受吗？
#
# Q5.【分年度】组合在哪些年份亏钱？这些年份商品市场大致是什么行情？
#
# Q6.【尽调】一个管理人说“我们交易 40 多个品种，风险很分散”。
#     你会让他提供什么数据、或者从估值表里看什么，来验证他是不是真的分散？
#
# Q7.【换手与成本】对比“等权(1倍名义)”“等权(10%波动)”“波动率调仓”三者的年化换手和年化成本。
#     波动率调仓的信号和等权完全一样，为什么换手更高？多出来的换手是“有用的交易”吗？
#     你能想到什么办法减少这部分换手？（提示：不是每次权重变一点点都需要调仓）
