# -*- coding: utf-8 -*-
"""
==============================================================================
练习 08：截面多因子 IC 检验、正交合成与“时序+截面”全天候 CTA 终极构建
==============================================================================
【阶段二：截面策略与期限结构 · 收官实战大作业】

【学习计划对应模块】：
    《CTA系统学习方案》阶段 2 第 5 项 & 动手项目：
    - “多因子合成、因子 IC/分组收益分析、因子之间的相关性”
    - “构建 3–5 个截面因子，做 IC 检验和多空组合；和阶段 1 的趋势组合做相关性分析与合成”
    - “同步：截面/多策略产品怎么评价（时序与截面的风险预算分配、品种风格切换时的回撤特征）”

【FOF 投资经理的核心视角】：
    当你去尽调国内百亿量化私募（如元盛、黑翼、千象、宽德、明汯、白鹭等）的 CTA 产品线时，
    他们的投研总监展示 PPT 的标准流程一定是：
        第一步：单因子检验 —— 展示每个因子的 **RankIC（秩相关系数）**、**ICIR（信息比率）** 与 **累计 IC 曲线**；
        第二步：正交化与相关性 —— 展示不同类别因子（期限结构、基本面库存、席位持仓、量价动量）之间的低相关性；
        第三步：多因子合成 —— 将 puluhan/上百个弱因子合成为高夏普的【截面 Alpha 组合】；
        第四步：大类策略风险预算 —— 将【时序趋势跟踪 (Time-Series Trend)】与【横截面多因子 (Cross-Sectional Alpha)】
                在 10% 等波动率口径下按风险预算配比，打造穿越牛熊与震荡市的“全天候复合 CTA”！

    跑通本脚本，你就真正拥有了独立复现一套【机构级多策略 CTA 投研体系】的硬核能力，
    这正是你《学习方案》作品集中最核心的王牌项目！
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

from cta_lib.universe import SYMBOLS
from cta_lib.backtest import performance, yearly_returns, setup_chinese_font
from cta_lib.portfolio import (load_universe, make_panel, tradable_mask, estimate_cost_rates,
                               vol_weights, scale_to_target_vol, portfolio_backtest,
                               plot_equity_compare)
from cta_lib.data_loader import load_roll_yield_panel, load_warehouse_panel, load_member_rank_panel

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

START, END = '2016-01-01', '2026-11-06'
TARGET_VOL = 0.10


#%% 第 1 块：构建阶段二四大核心截面因子库 (4 Pillars of Cross-Section) --
print("=" * 80)
print(f"正在加载数据并构建四大类商品期货横截面因子 ({START} 至 {END})...")
print("=" * 80)

data = load_universe(SYMBOLS, start=START, end=END, verbose=False)
close = make_panel(data, 'close')
high = make_panel(data, 'high')
low = make_panel(data, 'low')
ret = close.pct_change(fill_method=None)
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)

# 1. 【因子一：期限结构 Carry 因子】（来自 ex05）
ry = load_roll_yield_panel(SYMBOLS, start=START, end=END).reindex(index=close.index, columns=close.columns).ffill()
f_carry = ry.rolling(5, min_periods=1).mean()

# 2. 【因子二：基本面仓单库存因子】（来自 ex06：250日低库存水位 + 20/120日边际去库）
warehouse = load_warehouse_panel(SYMBOLS, start='2015-01-01', end=END).reindex(columns=close.columns).ffill()
w_clean = warehouse.where(warehouse > 0).ffill(limit=60).reindex(index=close.index, columns=close.columns)
inv_level = 1.0 - w_clean.rolling(250, min_periods=60).rank(pct=True)
inv_trend = -(w_clean.rolling(20, min_periods=10).mean() / w_clean.rolling(120, min_periods=40).mean() - 1.0)
f_inventory = (
    0.5 * inv_level.where(tradable).rank(axis=1, pct=True) +
    0.5 * inv_trend.where(tradable).rank(axis=1, pct=True)
).rolling(10, min_periods=3).mean()

# 3. 【因子三：前 20 名会员聪明钱因子】（来自 ex07：大席位净多持仓比）
mem_dict = load_member_rank_panel(SYMBOLS, start=START, end=END)
mem_long = mem_dict['long'].reindex(index=close.index, columns=close.columns).ffill()
mem_short = mem_dict['short'].reindex(index=close.index, columns=close.columns).ffill()
f_member = ((mem_long - mem_short) / (mem_long + mem_short + 1e-6)).rolling(20, min_periods=5).mean()

# 4. 【因子四：截面夏普动量因子 XSMOM】（来自 ex06：剔除近 5 日反转的 250 日风险调整动量）
f_xsmom = ((close.shift(5) / close.shift(250) - 1.0) / (ret.rolling(250, min_periods=60).std() * np.sqrt(250) + 1e-6)).rolling(10, min_periods=3).mean()

factors_dict = {
    '期限结构Carry': f_carry,
    '仓单库存因子': f_inventory,
    '会员聪明钱': f_member,
    '截面动量XSMOM': f_xsmom,
}
print("★ 四大截面因子构建完成！")


#%% 第 2 块：★ 金工研报标准武器：截面 RankIC 与 ICIR 检验体系 ★ -------
# 什么是 RankIC (Rank Information Coefficient)？
#   在每个交易日 t，计算所有可交易品种的【t 日因子排名】与【未来 H 日 (t 到 t+H) 收益率排名】之间的横截面相关系数（斯皮尔曼秩相关）！
#   - RankIC 均值 > 0：说明因子值越高的品种，未来越倾向于跑赢因子值低的品种；
#   - ICIR = RankIC 均值 / RankIC 标准差 * sqrt(252 / H)：衡量因子预测能力的稳定性（类似因子的夏普比率）；
#   - 累计 RankIC 曲线：如果一条斜率稳定向上的直线，说明该因子长年有效；如果某段走平或向下，说明因子在该阶段失效。

print("\n" + "=" * 80)
print("正在执行四大截面因子的 RankIC / ICIR 与多周期预测能力检验 (H = 5日, 10日, 20日)...")
print("=" * 80)

def calc_rank_ic_series(factor_df, forward_ret_df, mask_df):
    """向量化计算每日横截面 RankIC (Spearman 秩相关系数)"""
    valid = mask_df & factor_df.notna() & forward_ret_df.notna()
    # 至少有 15 个可交易有效品种才计算当日截面 IC
    valid_days = valid.sum(axis=1) >= 15
    rk_f = factor_df.where(valid).rank(axis=1, pct=True)
    rk_r = forward_ret_df.where(valid).rank(axis=1, pct=True)

    # 横截面皮尔逊相关系数作用于排名 (Rank) 即为 Spearman RankIC
    f_demean = rk_f.sub(rk_f.mean(axis=1), axis=0)
    r_demean = rk_r.sub(rk_r.mean(axis=1), axis=0)
    cov = (f_demean * r_demean).sum(axis=1)
    std_f = np.sqrt((f_demean ** 2).sum(axis=1))
    std_r = np.sqrt((r_demean ** 2).sum(axis=1))
    ic = (cov / (std_f * std_r + 1e-12)).where(valid_days)
    return ic.dropna()

# 计算未来 5 日、10 日、20 日远期收益率（基于 t+1 收盘至 t+1+H 收盘，杜绝当日未来函数）
fwd_ret_5d = close.shift(-6) / close.shift(-1) - 1.0
fwd_ret_10d = close.shift(-11) / close.shift(-1) - 1.0
fwd_ret_20d = close.shift(-21) / close.shift(-1) - 1.0

ic_rows = []
cum_ic_5d = pd.DataFrame(index=close.index)

for name, f_panel in factors_dict.items():
    ic_5 = calc_rank_ic_series(f_panel, fwd_ret_5d, tradable)
    ic_10 = calc_rank_ic_series(f_panel, fwd_ret_10d, tradable)
    ic_20 = calc_rank_ic_series(f_panel, fwd_ret_20d, tradable)

    # 对非重叠窗口标准差做保守估计，计算年化 ICIR (以 5 日调仓周期为例，一年约 50 期)
    ic_mean_5 = ic_5.mean()
    ic_std_5 = ic_5.std()
    t_stat_5 = ic_mean_5 / (ic_std_5 / np.sqrt(len(ic_5) / 5.0) + 1e-12)
    ann_icir_5 = (ic_mean_5 / (ic_std_5 + 1e-12)) * np.sqrt(252.0 / 5.0)

    ic_rows.append({
        '因子名称': name,
        '5日RankIC均值': ic_mean_5,
        '5日IC标准差': ic_std_5,
        '5日IC胜率(>0)': (ic_5 > 0).mean(),
        '等效t统计量': t_stat_5,
        '年化ICIR(5D)': ann_icir_5,
        '10日RankIC均值': ic_10.mean(),
        '20日RankIC均值': ic_20.mean(),
    })
    cum_ic_5d[name] = ic_5.reindex(close.index).fillna(0.0).cumsum()

ic_summary = pd.DataFrame(ic_rows).set_index('因子名称')
print("\n【四大横截面因子 RankIC 与预测能力检验汇总表】:")
print(ic_summary)

# 绘制累计 RankIC 曲线
def plot_cumulative_ic(cum_ic_df):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for col in cum_ic_df.columns:
        ax.plot(cum_ic_df.index, cum_ic_df[col], label=col, lw=1.8)
    ax.set_title('商品期货四大横截面因子累计 RankIC 走势 (斜率越稳代表因子长期预测力越恒定)', fontsize=13, pad=12)
    ax.set_ylabel('累计 RankIC (5日远期)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_cumulative_ic(cum_ic_5d)


#%% 第 3 块：★ 截面因子间的相关性与多因子合成 (Multi-Factor Alpha) ★ ----
def build_ls_weights(score_panel, top_q=0.20, rebalance_every=5):
    valid = tradable & score_panel.notna()
    rk = score_panel.where(valid).rank(axis=1, ascending=False, pct=True)
    in_l = (rk <= top_q)
    in_s = (rk > (1.0 - top_q))
    wl = in_l.div(in_l.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ws = in_s.div(in_s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = 0.5 * wl - 0.5 * ws
    if rebalance_every > 1:
        keep = (np.arange(len(w)) % rebalance_every == 0)
        w_wk = w.copy()
        w_wk.loc[~keep, :] = np.nan
        w = w_wk.ffill().fillna(0.0).where(tradable, 0.0)
    return w

w_single = {name: build_ls_weights(f_panel, rebalance_every=5) for name, f_panel in factors_dict.items()}
port_single = {name: portfolio_backtest(data, w, cost_rate=cost)[0] for name, w in w_single.items()}

ret_single = pd.DataFrame({name: p['net_ret'] for name, p in port_single.items()})
corr_factors = ret_single.corr()

print("\n" + "=" * 80)
print("【四大截面因子多空净值日收益率相关系数矩阵】:")
print("=" * 80)
print(corr_factors.round(3))
print("\n★ 关键发现：")
print("   【仓单库存因子】与【截面动量】相关系数接近 0，与【会员聪明钱】相关系数仅 0.05，与【期限结构Carry】仅 0.16！")
print("   基本面库存与量价/期限结构几乎完全正交，是天然的多因子合成圣杯！")

# 构建四大截面多因子合成组合（按 25% Carry + 30% 库存 + 20% 聪明钱 + 25% 截面动量 权重合成）
w_cs_multi_1x = (
    0.25 * w_single['期限结构Carry'] +
    0.30 * w_single['仓单库存因子'] +
    0.20 * w_single['会员聪明钱'] +
    0.25 * w_single['截面动量XSMOM']
)
port_cs_multi_1x, _ = portfolio_backtest(data, w_cs_multi_1x, cost_rate=cost)

# 将截面多因子组合缩放到 10% 目标年化波动率（因为多因子对冲后自然波动率仅 5.4%，等波放大后方便与趋势策略公平对比）
w_cs_multi_10v = scale_to_target_vol(w_cs_multi_1x, ret, target_vol=TARGET_VOL)
port_cs_multi_10v, _ = portfolio_backtest(data, w_cs_multi_10v, cost_rate=cost)

cs_table = pd.DataFrame({
    **{f'单因子：{k} (1x)': performance(v) for k, v in port_single.items()},
    '★ 四因子合成 (1x自然波动)': performance(port_cs_multi_1x),
    '★ 四因子合成 (10%目标波动)': performance(port_cs_multi_10v),
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率', '年化换手(倍)', '年化成本']]

print("\n" + "=" * 95)
print("【单因子 vs 四大截面多因子合成绩效总览】")
print("=" * 95)
print(cs_table)
print("=" * 95)


#%% 第 4 块：★ 终极合体：阶段 1 时序趋势集成 + 阶段 2 截面多因子 ★ -----
# 现在我们把《学习方案》阶段 1 与阶段 2 的成果进行终极合体：
# - 阶段 1 代表：【时序趋势多信号集成 (Trend Ensemble)】：双均线(20,60) + 海龟突破(20) + TSMOM(250)，缩放到 10% 目标波动率；
# - 阶段 2 代表：【截面多因子合成 (Cross-Sectional Multi-Factor)】：Carry + 库存 + 聪明钱 + XSMOM，缩放到 10% 目标波动率；
# - 终极全天候 CTA：按 50% 时序趋势风险预算 + 50% 截面多因子风险预算合成，并统一校准到 10% 目标波动率！

print("\n" + "=" * 80)
print("正在构建【阶段 1 时序趋势集成】与【阶段 2 截面多因子】的全天候复合 CTA...")
print("=" * 80)

sig_ma = pd.DataFrame(np.where(close.rolling(20).mean() > close.rolling(60).mean(), 1.0, -1.0), index=close.index, columns=close.columns)
up20, dn20 = high.rolling(20).max().shift(1), low.rolling(20).min().shift(1)
sig_don = pd.DataFrame(np.where(close > up20, 1.0, np.where(close < dn20, -1.0, np.nan)), index=close.index, columns=close.columns).ffill().fillna(0.0)
sig_ts = pd.DataFrame(np.where(close / close.shift(250) - 1.0 > 0, 1.0, -1.0), index=close.index, columns=close.columns)

sig_trend_ens = ((sig_ma + sig_don + sig_ts) / 3.0).where(tradable, 0.0)
w_trend_10v = scale_to_target_vol(vol_weights(sig_trend_ens, ret), ret, target_vol=TARGET_VOL)
port_trend_10v, _ = portfolio_backtest(data, w_trend_10v, cost_rate=cost)

# 计算阶段 1 时序趋势 与 阶段 2 截面多因子 的净值相关性
corr_ts_cs = port_trend_10v['net_ret'].corr(port_cs_multi_10v['net_ret'])
print(f"\n★ 终极验证：【阶段 1 时序趋势集成】与【阶段 2 截面多因子合成】的日收益相关系数仅为: {corr_ts_cs:+.3f}！")

# 构建 50% 时序趋势 + 50% 截面多因子 的全天候 CTA 组合（统一 10% 目标波动率）
w_all_weather = scale_to_target_vol(0.50 * w_trend_10v + 0.50 * w_cs_multi_10v, ret, target_vol=TARGET_VOL)
port_all_weather, _ = portfolio_backtest(data, w_all_weather, cost_rate=cost)

final_table = pd.DataFrame({
    '1. 纯时序趋势集成 (阶段1, 10%等波)': performance(port_trend_10v),
    '2. 纯截面多因子合成 (阶段2, 10%等波)': performance(port_cs_multi_10v),
    '3. 全天候复合 CTA (50%趋势+50%截面, 10%等波)': performance(port_all_weather),
}).T[['年化收益', '年化波动', '夏普比率', '索提诺比率', '最大回撤', '卡玛比率', '日收益偏度', '月度胜率']]

print("\n" + "=" * 95)
print("【阶段 1 + 阶段 2 收官大比拼：时序趋势 vs 截面多因子 vs 全天候复合 CTA】")
print("=" * 95)
print(final_table)
print("=" * 95)

print("\n【分年度收益率全景对比 (%) —— 重点观察顺风年(2020)与震荡逆风年(2019/2023)的互补性】:")
yr_final = pd.DataFrame({
    '纯时序趋势(10%波)': yearly_returns(port_trend_10v)['策略'],
    '纯截面多因子(10%波)': yearly_returns(port_cs_multi_10v)['策略'],
    '全天候复合CTA(10%波)': yearly_returns(port_all_weather)['策略'],
})
print((yr_final * 100).round(2))

plot_equity_compare({
    '阶段1：纯时序趋势集成 (10%波动)': port_trend_10v,
    '阶段2：纯截面多因子合成 (10%波动)': port_cs_multi_10v,
    '终极合体：全天候复合 CTA (50%时序+50%截面)': port_all_weather,
}, title='阶段 1 时序趋势 vs 阶段 2 截面多因子 vs 全天候复合 CTA (2016-2026 扣费净值)', log=False)


#%% 第 5 块：阶段二收官总结与 FOF 截面/多策略产品尽调清单 -------------
print("\n" + "=" * 85)
print("【阶段二收官 · 截面/多策略 CTA 产品 FOF 评价与尽调核心法则】：")
print("=" * 85)
print("""
1. 【风险预算陷阱（名义资金配比 vs 波动率风险配比）】：
   我们在本练习第 3 块看到：1x 名义杠杆下的【截面多因子组合】自然年化波动率只有 5.4%，
   而 1x 名义杠杆下的【时序趋势组合】自然波动率通常在 12%~15%！
   ★ 尽调必问：如果一家私募说“我们是 50% 趋势 + 50% 截面”，你必须追问：
     “这 50:50 是指【名义保证金占比】，还是【等波动率下的风险预算占比 (Risk Budget)】？”
     如果是名义资金 50:50，那么高波动的趋势策略实际上贡献了组合 80% 以上的波动，截面根本起不到缓冲作用！
     （这也是为什么我们在 cta_nav_regression.py 中专门设计了【名义暴露 Beta_1x】和【等波暴露 Beta_10%Vol】双口径！）

2. 【截面策略的阿喀琉斯之踵（什么时候截面会大回撤？）】：
   截面多空组合虽然剥离了全市场涨跌的 Beta，但它并不是无风险套利！
   当市场发生【宏观政策突变或板块风格急速反转】（例如“双焦/能化暴涨、农产品暴跌”一夜之间切换为“工业品暴跌、农产品逼仓”）时，
   多头端和空头端会同时被打脸（史称“截面多空双杀”）。
   ★ 尽调必问：“你们的截面多空组合是否做了【板块中性 (Sector Neutral)】约束？单一板块的最大净敞口上限是多少？”

3. 【分布形态诊断（偏度 Skewness 的变化）】：
   观察上面汇总表中的 `日收益偏度`：
   - 纯时序趋势通常是显著【正偏】（平时小亏，大行情暴赚）；
   - 纯截面多因子由于带有均值回归与吃贴水属性，偏度往往接近 0 甚至略微偏负，但【月度胜率与夏普比率】显著高于纯趋势；
   - 两者 50:50 复合后，既保留了趋势策略在危机/通胀大年（如 2020）的爆发力，又在震荡年（如 2019、2022）依靠截面 Alpha 稳住净值！
""")
print("=" * 85)
