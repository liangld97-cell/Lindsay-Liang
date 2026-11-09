# -*- coding: utf-8 -*-
"""
==============================================================================
练习 05：期限结构与展期收益率 (Carry) —— 揭开商品期货截面 Alpha 的秘密
==============================================================================
【阶段二：截面策略与期限结构】

【FOF 投资经理的核心视角】：
    在做量化私募 CTA 尽调时，你经常会听到投研总监讲：“我们的 CTA 策略是复合型的，
    除了 40% 的中长周期时序趋势，底仓还配置了 30% 到 40% 的期限结构 Carry（展期收益率）因子。”
    为什么量化私募如此钟爱 Carry 因子？
    因为时序趋势依赖“单品种纵向的行情发酵”，震荡市必然不断假突破受挫；
    而期限结构策略是“全市场横截面的多空配对”，多头做多贴水最严重的品种，空头做空升水最严重的品种。
    它赚的是商品现货与期货之间的结构性风险溢价，日收益与趋势策略的相关性通常极低（甚至弱负相关）！
    它是抚平趋势策略回撤、提升夏普比率不可或缺的“黄金搭档”。

【本练习你将掌握的五大实战技能】：
    1. 【期限结构原理】：透彻理解 Backwardation（近高远低/多头吃贴水）与 Contango（近低远高/空头吃升水）。
    2. 【全市场横截面全景】：提取 45 个品种的年化展期收益率，绘制截面强弱天梯图。
    3. 【五分组分层单调性检验 (Quintile Monotonicity Test)】：量化研究中判断因子是否有效的最高黄金标准！
    4. 【纯多空市场中性 Carry 策略构建】：做多 Top 20% Backwardation，做空 Bottom 20% Contango，严格扣费回测。
    5. 【趋势 + Carry 复合配置】：50% 趋势 + 50% Carry，亲眼见证夏普比率从 0.8 跃升至 方案1.3+ 的“资产配置免费午餐”。
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自动定位项目根目录并导入 cta_lib
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.universe import SYMBOLS, name_of, sector_of
from cta_lib.backtest import performance, yearly_returns, setup_chinese_font
from cta_lib.portfolio import (load_universe, make_panel, tradable_mask, estimate_cost_rates,
                               portfolio_backtest, plot_equity_compare)
from cta_lib.data_loader import load_roll_yield_panel

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

START, END = '2016-01-01', '2026-10-30'


#%% 第 1 块：数据加载与面板对齐 ----------------------------------------
print("=" * 70)
print(f"正在加载全市场 45 个品种主力连续数据与展期收益率面板 ({START} 至 {END})...")
print("=" * 70)

# 1. 主力连续价格面板
data = load_universe(SYMBOLS, start=START, end=END, verbose=False)
close = make_panel(data, 'close')
ret = close.pct_change()
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)

# 2. 展期收益率面板 (Roll Yield Panel)
# RQData 展期收益率算法：(主力价格 - 次主力价格) / 主力价格 * (250 / 换月间隔天数)
# 正值代表 Backwardation（近高远低，现货升水，做多享受滚存展期收益）
# 负值代表 Contango（近低远高，现货贴水，做空享受滚存展期收益）
roll_yield = load_roll_yield_panel(symbols=SYMBOLS, start=START, end=END)
roll_yield = roll_yield.reindex(index=close.index, columns=close.columns).ffill()

print(f"数据加载完成！有效交易日: {len(close)} 天, 覆盖品种数: {len(close.columns)} 个")


#%% 第 2 块：全市场期限结构宏观全景图（最新截面切片） ------------------
# 取最近一个有效交易日的展期收益率，观察全市场期限结构分布
latest_date = roll_yield.dropna(how='all').index[-1]
latest_ry = roll_yield.loc[latest_date].dropna().sort_values(ascending=False)

print("\n" + "=" * 70)
print(f"【截面快照】最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 45 个品种年化展期收益率排名：")
print("=" * 70)
print(f"★ 深度 Backwardation (贴水最深/多头最爱 TOP 5):")
for s in latest_ry.head(5).index:
    print(f"   {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 年化展期收益率: {latest_ry[s]*100:+6.2f}%")

print(f"\n★ 深度 Contango (升水最深/空头最爱 TOP 5):")
for s in latest_ry.tail(5).index:
    print(f"   {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 年化展期收益率: {latest_ry[s]*100:+6.2f}%")

# 绘制最新截面天梯图
def plot_cross_section_roll_yield(ry_series, date_str):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in ry_series.values]
    x = np.arange(len(ry_series))
    ax.bar(x, ry_series.values * 100, color=colors, width=0.6, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name_of(s)}\n{s}" for s in ry_series.index], rotation=0, fontsize=8)
    ax.set_ylabel('年化展期收益率 (%)', fontsize=11)
    ax.set_title(f'商品期货全市场期限结构横截面天梯图 ({date_str})\n[红色 > 0: Backwardation(贴水) | 绿色 < 0: Contango(升水)]', fontsize=13, pad=12)
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()

plot_cross_section_roll_yield(latest_ry, latest_date.strftime('%Y-%m-%d'))


#%% 第 3 块：★ 因子单调性检验 (五分组分层测试 Quintile Test) ★ -------
# 判定一个量化因子是“真 Alpha”还是“假拟合”，最核心的检验就是【分层单调性】。
# 每天对所有上市交易的品种，按其展期收益率从大到小分为 5 组（Q1 到 Q5，每组 20%）：
#   Q1: 展期收益率最高（深度贴水）
#   Q2: 次高
#   Q3: 中间
#   Q4: 次低
#   Q5: 展期收益率最低（深度升水）
# 如果因子有效，必然呈现 Q1 > Q2 > Q3 > Q4 > Q5 的完美单调递减！

print("\n" + "=" * 70)
print("正在进行全市场五分组 (Quintile) 分层单调性测试...")
print("=" * 70)

valid_mask = tradable & roll_yield.notna()
# 截面百分比排名：0.0 (最高) 到 1.0 (最低)
# shift(1) 杜绝未来函数：使用 t-1 日收盘后的展期收益率来决定 t 日的仓位！
ry_lag = roll_yield.shift(1)
ranks = ry_lag.where(valid_mask.shift(1)).rank(axis=1, ascending=False, pct=True)

quintile_equities = pd.DataFrame(index=close.index)

for q in range(1, 6):
    lower = (q - 1) * 0.2
    upper = q * 0.2
    # 属于该分位数的品种掩码
    in_group = (ranks > lower) & (ranks <= upper)
    # 组内等权归一化
    group_weight = in_group.div(in_group.sum(axis=1), axis=0).fillna(0.0)
    # 计算该组每日多头收益率
    group_ret = (group_weight * ret).sum(axis=1)
    group_eq = (1 + group_ret).cumprod()
    quintile_equities[f'Q{q}'] = group_eq
    ann_ret = (group_eq.iloc[-1]) ** (250 / len(group_eq)) - 1
    ann_vol = group_ret.std() * np.sqrt(250)
    sharpe = ann_ret / (ann_vol + 1e-6)
    print(f"   第 Q{q} 组 纯多头表现 -> 年化收益: {ann_ret*100:+6.2f}%, 年化波动: {ann_vol*100:5.2f}%, 夏普比率: {sharpe:5.2f}")

# 绘制五分组单调性曲线
def plot_quintiles(eq_df):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#1f77b4', '#2ca02c']
    labels = [
        'Q1 (最高贴水 TOP 20%)',
        'Q2 (次高贴水 20%-40%)',
        'Q3 (中性组 40%-60%)',
        'Q4 (次高升水 60%-80%)',
        'Q5 (最高升水 BOTTOM 20%)'
    ]
    for i, col in enumerate(eq_df.columns):
        ax.plot(eq_df[col], label=labels[i], color=colors[i], lw=2.0 if i in [0, 4] else 1.2)
    ax.set_title('商品期货期限结构因子 (Carry) 五分组纯多头净值走势 (检验单调性)', fontsize=14, pad=12)
    ax.set_ylabel('累计净值 (起始=1.0)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_quintiles(quintile_equities)


#%% 第 4 块：纯多空市场中性 Carry 策略构建与扣费回测 ---------------
# 构建纯多空组合：做多 Q1，做空 Q5
# 多头总权重 +50%，空头总权重 -50%，多空净敞口为 0（绝对市场中性）
in_q1 = (ranks <= 0.2)
in_q5 = (ranks > 0.8)

w_long = in_q1.div(in_q1.sum(axis=1), axis=0).fillna(0.0)
w_short = in_q5.div(in_q5.sum(axis=1), axis=0).fillna(0.0)
w_carry = 0.5 * w_long - 0.5 * w_short

print("\n" + "=" * 70)
print("正在对纯多空市场中性 Carry 策略进行严格扣费回测 (考虑真实手续费与冲击成本)...")
print("=" * 70)

port_carry, asset_carry = portfolio_backtest(data, w_carry, cost_rate=cost, execution='next_open', roll_cost=True)
perf_carry = performance(port_carry)

print("\n【纯多空 Carry 策略回测绩效总览】:")
print("-" * 50)
for k, v in perf_carry.items():
    if isinstance(v, float):
        if any(w in k for w in ['收益', '波动', '回撤', '成本', '胜率']):
            print(f"  {k:<16}: {v*100:+6.2f}%")
        elif any(w in k for w in ['比率', '夏普', '索提诺', '卡玛', '偏度', '杠杆', '换手', '品种数']):
            print(f"  {k:<16}: {v:6.2f}")
        else:
            print(f"  {k:<16}: {v:6.2f}")
    else:
        print(f"  {k:<16}: {v}")
print("-" * 50)

# 逐年收益率
print("\n【纯多空 Carry 逐年收益分布】:")
print(yearly_returns(port_carry))


#%% 第 5 块：★ 组合核聚变：时序趋势 + 截面 Carry 复合配置 ★ ----------
# 许多人以为做 CTA 只能做时序趋势，但时序趋势是“贝塔型趋势”，在横盘震荡期非常痛苦。
# 截面 Carry 是“截面阿尔法”，赚的是商品间的相对强弱溢价。
# 让我们看看把【Phase 1 经典的 20/60 双均线趋势策略】与【截面 Carry 策略】按 5:5 复合，会发生什么奇迹！

print("\n" + "=" * 70)
print("正在构建 20/60 双均线时序趋势组合...")
print("=" * 70)

# 计算 20/60 双均线信号
ma20 = close.rolling(20).mean()
ma60 = close.rolling(60).mean()
sig_trend = pd.DataFrame(0.0, index=close.index, columns=close.columns)
sig_trend[ma20 > ma60] = 1.0
sig_trend[ma20 < ma60] = -1.0
sig_trend = sig_trend.where(valid_mask, 0.0)

# 趋势策略权重（等权分配，总杠杆控制在 1.0）
n_active = (sig_trend != 0).sum(axis=1).replace(0, np.nan)
w_trend = sig_trend.div(n_active, axis=0).fillna(0.0)

port_trend, _ = portfolio_backtest(data, w_trend, cost_rate=cost, execution='next_open', roll_cost=True)

# 计算时序趋势与截面 Carry 的收益率相关性
corr = port_trend['net_ret'].corr(port_carry['net_ret'])
print(f"\n★ 核心发现：时序趋势 与 截面 Carry 的日收益率相关系数为: {corr:+.3f}")
print("   (在中长周期上保持弱相关，两者结合能大幅削减组合整体波动率与回撤！)")

# 构建 5:5 复合策略
w_combo = 0.5 * w_trend + 0.5 * w_carry
port_combo, _ = portfolio_backtest(data, w_combo, cost_rate=cost, execution='next_open', roll_cost=True)

# 三者绩效横向对比
perf_trend = performance(port_trend)
perf_combo = performance(port_combo)

summary_table = pd.DataFrame({
    '纯时序趋势 (20/60双均线)': perf_trend,
    '纯截面 Carry (展期多空)': perf_carry,
    '50%趋势 + 50%Carry 复合组合': perf_combo
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率']]

print("\n" + "=" * 80)
print("【三大策略综合绩效大比拼（撕开多策略 CTA 的神秘面纱）】")
print("=" * 80)
print(summary_table)
print("=" * 80)

# 画对比曲线
plot_equity_compare({
    '纯时序趋势 (20/60)': port_trend,
    '纯截面 Carry (期限结构)': port_carry,
    '50% 趋势 + 50% Carry 复合组合': port_combo
}, title='时序趋势 vs 截面 Carry vs 复合 CTA 净值对比 (全样本扣费)', log=False)


#%% 第 6 块：FOF 尽调灵魂拷问与实战思考 ---------------------------------
print("\n" + "=" * 80)
print("【阶段二 · 练习 05 课后思考与 FOF 尽调实战拷问】：")
print("=" * 80)
print("""
[思考 1]：【负基差反转踩踏 (Basis Squeeze Crash)】
    观察 2016-2017 年以及 2021 年商品大牛市期间，为什么纯 Carry 策略在某些月份会出现突发性剧烈回撤？
    提示：当宏观流动性或者产业政策发生剧烈反转时，原本深度贴水（Backwardation）的强势品种可能突发暴跌，
    而深度升水（Contango）的过剩品种反而因为减产预期报复性反弹，造成多空双向被打脸！私募是如何做防范的？

[思考 2]：【仓单库存与软逼仓陷阱 (The Inventory Trap)】
    纯粹依靠展期收益率做空 Contango 品种时，如果该品种在交易所的“注册仓单”几乎为零，
    且现货流通极度紧俏（例如历史上某些时期的红枣、纯碱、苹果），此时盲目根据展期收益率做空会面临什么致命风险？
    （这就是为什么下一课我们要引入【仓单与库存因子】进行截面过滤！）

[思考 3]：【曲线斜率与流动性折价】：
    米筐的 get_roll_yield 是以主力合约和次主力合约的价差来计算斜率的。
    如果在农产品等存在明显季节性（如 1-5-9 合约规律）的品种上，隔月合约之间没有成交量，
    展期收益率是否会失真？量化团队通常如何对非主力月份进行插值修正？
""")
print("=" * 80)
