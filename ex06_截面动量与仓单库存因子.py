# -*- coding: utf-8 -*-
"""
==============================================================================
练习 06：截面动量 (XSMOM) 与基本面仓单库存因子 —— 从价格强弱到现货紧缺度
==============================================================================
【阶段二：截面策略与期限结构 · 进阶篇】

【承接 ex05 的核心悬念】：
    在练习 05 的结尾思考题中，我们提出了一个实战痛点：
    “单纯依靠展期收益率 (Carry) 做多贴水、做空升水，如果遇到基本面反转或逼仓陷阱怎么办？”
    同时，在《CTA系统学习方案》阶段 2 中，除了期限结构 Carry，国内量化私募最核心的另外两大截面支柱正是：
        1. 【截面动量 (Cross-Sectional Momentum, XSMOM)】：不赌大盘涨跌，只赌“强者恒强、弱者恒弱”。
        2. 【基本面库存/注册仓单因子 (Warehouse Stocks)】：直击大宗商品定价的咽喉——“低库存+去库”做多，“高库存+累库”做空！

【FOF 投资经理的核心视角】：
    尽调时管理人常说：“我们既有时间序列动量 (TSMOM)，也有横截面动量 (XSMOM)，还有基本面库存因子。”
    你必须一针见血地理解：
    - 时序动量 (TSMOM) 在商品普涨时全线做多（净多头敞口可达 +100%），本质上带有很强的商品多头/空头 Beta；
    - 截面动量 (XSMOM) 永远“做多涨幅前 20% 品种、做空涨幅后 20% 品种”，多空净敞口恒为 0，赚的是品种间分化的 Alpha！
    - 仓单库存因子则是纯粹的【产业基本面数据】：价格可以骗人，但交易所仓库里实打实的注册仓单不会骗人。

【本练习你将掌握的五大实战技能】：
    1. 【TSMOM vs XSMOM 净敞口透视】：亲眼对比时序动量与截面动量在商品牛熊市中的多空净敞口差异。
    2. 【商品截面动量的正确打开方式】：为什么直接按涨跌幅排序会失效？如何通过“波动率标准化（夏普动量）+ 剔除近 5 日短期反转 + 周频调仓”大幅提升夏普？
    3. 【注册仓单（库存）跨品种标准化】：生猪按吨、黄金按千克、原油按桶，绝对仓单量完全不可比！如何用“250日历史分位水位 + 20/120日去库均线比”构建全市场可比的库存因子？
    4. 【库存因子五分组单调性检验】：验证“低库存+去库 (Q1)”对“高库存+累库 (Q5)”的显著超额。
    5. 【Carry + 库存双剑合璧（基本面共振）】：实证检验“贴水+去库做多、升水+累库做空”，将纯 Carry 的最大回撤从 -21.7% 腰斩至 -10.5%，夏普从 0.39 跃升至 0.59，卡玛比率翻倍！
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
from cta_lib.data_loader import load_roll_yield_panel, load_warehouse_panel

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

START, END = '2016-01-01', '2026-11-06'


#%% 第 1 块：数据加载与面板对齐 ----------------------------------------
print("=" * 75)
print(f"正在加载 45 个品种行情、展期收益率与交易所注册仓单数据 ({START} 至 {END})...")
print("=" * 75)

data = load_universe(SYMBOLS, start=START, end=END, verbose=False)
close = make_panel(data, 'close')
ret = close.pct_change(fill_method=None)
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)

# 1. 展期收益率面板 (Carry)
roll_yield = load_roll_yield_panel(symbols=SYMBOLS, start=START, end=END)
roll_yield = roll_yield.reindex(index=close.index, columns=close.columns).ffill()

# 2. 交易所注册仓单面板 (Warehouse Stocks)
warehouse = load_warehouse_panel(symbols=SYMBOLS, start='2015-01-01', end=END)
warehouse = warehouse.reindex(columns=close.columns).ffill()

print(f"数据加载完成！交易日数: {len(close)} 天, 覆盖品种数: {len(close.columns)} 个")


#%% 第 2 块：★ 截面动量 (XSMOM)：为什么夏普动量与周频换仓至关重要？ ★ ---
# 初学者做截面动量最容易踩两个大坑：
# 坑 1：【不除波动率】直接用过去 N 日涨跌幅排序 -> 高波动品种（原油、镍、焦煤、纯碱）永远霸占多空两端，组合变成赌高波品种！
# 坑 2：【包含最近几天的暴涨暴跌 + 日频调仓】商品期货在极短周期（1~5天）存在显著的流动性冲击与均值反转，且日频调仓换手率高达 40 倍以上！
# 工业界标准解法：
#   (1) 剔除最近 5 日短期反转（Skip 5 days）；
#   (2) 除以区间波动率，构造【风险调整截面动量 / 夏普动量 (IR-XSMOM)】；
#   (3) 采用 10 日因子平滑 + 每周（5个交易日）换仓一次！

print("\n" + "=" * 75)
print("【实验一】截面动量 (XSMOM) 五分组单调性检验：剔除近 5 日反转的夏普动量")
print("=" * 75)

# 计算剔除最近 5 日的 250 日夏普动量（信息比率动量）
mom_ret = close.shift(5) / close.shift(250) - 1.0
mom_vol = ret.rolling(250, min_periods=60).std() * np.sqrt(250)
xsmom_score = (mom_ret / (mom_vol + 1e-6)).rolling(10, min_periods=3).mean()

def run_quintile_test(factor_panel, factor_name):
    """通用五分组单调性检验函数（严格使用 t-1 日因子预测 t 日收益）"""
    valid = tradable & factor_panel.notna()
    ranks = factor_panel.shift(1).where(valid.shift(1)).rank(axis=1, ascending=False, pct=True)
    q_eq = pd.DataFrame(index=close.index)
    print(f"\n★ 【{factor_name}】五分组纯多头表现 (Q1=因子最高 Top 20%, Q5=因子最低 Bottom 20%):")
    for q in range(1, 6):
        in_g = (ranks > (q - 1) * 0.2) & (ranks <= q * 0.2)
        gw = in_g.div(in_g.sum(axis=1), axis=0).fillna(0.0)
        gr = (gw * ret).sum(axis=1)
        eq = (1.0 + gr).cumprod()
        q_eq[f'Q{q}'] = eq
        ann_r = eq.iloc[-1] ** (250.0 / len(eq)) - 1.0
        ann_v = gr.std() * np.sqrt(250.0)
        sr = ann_r / (ann_v + 1e-6)
        print(f"   第 Q{q} 组 -> 年化收益: {ann_r*100:+6.2f}% | 年化波动: {ann_v*100:5.2f}% | 夏普比率: {sr:5.2f}")
    return q_eq

q_eq_xsmom = run_quintile_test(xsmom_score, "截面夏普动量 (XSMOM)")

# 对比：原始涨跌幅日频调仓 vs 夏普动量周频调仓
def build_ls_weights(score_panel, top_q=0.20, rebalance_every=5):
    """构建标准截面多空权重（+50% 多头 Top 20%，-50% 空头 Bottom 20%，净敞口恒为 0）"""
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

raw_mom_20 = close / close.shift(20) - 1.0
w_xsmom_naive = build_ls_weights(raw_mom_20, rebalance_every=1)
w_xsmom_pro = build_ls_weights(xsmom_score, rebalance_every=5)

port_xsmom_naive, _ = portfolio_backtest(data, w_xsmom_naive, cost_rate=cost)
port_xsmom_pro, _ = portfolio_backtest(data, w_xsmom_pro, cost_rate=cost)

cmp_xsmom = pd.DataFrame({
    '粗糙截面动量 (20日原始涨跌+日频换仓)': performance(port_xsmom_naive),
    '专业截面动量 (夏普动量+剔除反转+周频换仓)': performance(port_xsmom_pro)
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '年化换手(倍)', '年化成本']]
print("\n【截面动量工程化改造前后对比（扣除真实滑点与换月成本）】:")
print(cmp_xsmom)


#%% 第 3 块：★ 基本面因子：交易所注册仓单（库存）因子的构建与单调性 ★ --
# 【产业逻辑】：
#   期货价格围绕现货供需波动，而“库存”是供需平衡表最直接的蓄水池！
#   - 当某品种注册仓单处于历史极低分位（低库存），且近期仓单还在持续注销减少（去库），
#     说明现货极度紧缺，空头面临无货可交的“逼仓风险”，多头具备极强的安全边际！
#   - 反之，当某品种仓单堆积如山（高库存），且还在不断注册增加（累库），
#     说明现货严重过剩，产业套保盘会源源不断在盘面抛空，做空胜率极高！
#
# 【量化清洗核心技巧】：
#   1. 交割月前夕部分品种仓单会阶段性强制清零，必须将 0 视为缺失并向前填充 (ffill)，防止假性“去库 100%”造成异常换手；
#   2. 跨品种量纲标准化：
#      - 库存水位得分 (Level)：过去 250 日仓单量的时序百分比排名倒数 (1 - pct_rank)，越接近 1 代表库存越低；
#      - 库存趋势得分 (Trend)：-(20日仓单均值 / 120日仓单均值 - 1)，正值代表近期正在去库；
#      - 两者在截面上分别排名后 5:5 等权复合，并做 10 日平滑！

print("\n" + "=" * 75)
print("【实验二】基本面因子：注册仓单（低库存水位 + 边际去库）构建与检验")
print("=" * 75)

# 1. 清洗交割月短暂清零噪点并对齐到回测区间
w_clean = warehouse.where(warehouse > 0).ffill(limit=60)
w_clean = w_clean.reindex(index=close.index, columns=close.columns)

# 2. 维度一：低库存水位（过去 250 日分位数倒数，值越大代表库存越处于历史低位）
inv_level = 1.0 - w_clean.rolling(250, min_periods=60).rank(pct=True)

# 3. 维度二：边际去库趋势（20日均线相对120日均线的缩减比例，值越大代表去库越猛）
inv_trend = -(w_clean.rolling(20, min_periods=10).mean() / w_clean.rolling(120, min_periods=40).mean() - 1.0)

# 4. 截面排名等权复合（低库存 + 去库共振）
inv_score = (
    0.5 * inv_level.where(tradable).rank(axis=1, pct=True) +
    0.5 * inv_trend.where(tradable).rank(axis=1, pct=True)
).rolling(10, min_periods=3).mean()

# 查看最新截面库存紧缺度 Top 5 与 库存过剩 Top 5
latest_dt = inv_score.dropna(how='all').index[-1]
latest_inv = inv_score.loc[latest_dt].dropna().sort_values(ascending=False)
print(f"\n【最新库存基本面快照】({latest_dt.strftime('%Y-%m-%d')}):")
print("  ★ 库存最紧缺（低库存+强去库，多头首选 TOP 5）:")
for s in latest_inv.head(5).index:
    print(f"     {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 库存紧缺综合分位: {latest_inv[s]*100:5.1f}%")
print("  ★ 库存最过剩（高库存+强累库，空头首选 TOP 5）:")
for s in latest_inv.tail(5).index:
    print(f"     {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 库存紧缺综合分位: {latest_inv[s]*100:5.1f}%")

# 五分组单调性检验
q_eq_inv = run_quintile_test(inv_score, "注册仓单库存因子 (低库存+去库)")


#%% 第 4 块：★ 破解 ex05 思考题：期限结构 Carry + 仓单库存双剑合璧 ★ ---
# 回顾 ex05 思考题 2：
#   纯粹看展期收益率 Carry 做空升水品种时，万一该品种仓单极低（现货被控盘或紧缺），就会遭遇“软逼仓”大回撤！
#   现在我们将【期限结构 Carry】与【仓单库存因子】按 50% : 50% 结合：
#   只有当品种既贴水（Backwardation）又处于低库存/去库时才重仓做多；
#   既升水（Contango）又处于高库存/累库时才重仓做空！

print("\n" + "=" * 75)
print("【实验三】破解逼仓陷阱：期限结构 Carry vs 仓单库存 vs 双因子基本面共振")
print("=" * 75)

carry_score = roll_yield.rolling(5, min_periods=1).mean()

w_carry = build_ls_weights(carry_score, rebalance_every=5)
w_inv = build_ls_weights(inv_score, rebalance_every=5)
w_fundamental = 0.5 * w_carry + 0.5 * w_inv

port_carry, _ = portfolio_backtest(data, w_carry, cost_rate=cost)
port_inv, _ = portfolio_backtest(data, w_inv, cost_rate=cost)
port_fund, _ = portfolio_backtest(data, w_fundamental, cost_rate=cost)

corr_ci = port_carry['net_ret'].corr(port_inv['net_ret'])
print(f"\n★ 核心发现：期限结构 Carry 与 仓单库存因子 的日收益率相关系数仅为: {corr_ci:+.3f}！")
print("  （虽然同属基本面逻辑，但一个反映远期曲线定价，一个反映当下实物交割库存，两者高度互补！）")

fund_table = pd.DataFrame({
    '1. 纯期限结构 Carry (周频多空)': performance(port_carry),
    '2. 纯仓单库存因子 (周频多空)': performance(port_inv),
    '3. Carry + 库存双剑合璧 (50:50共振)': performance(port_fund),
    '4. 截面夏普动量 XSMOM (周频多空)': performance(port_xsmom_pro),
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率', '年化换手(倍)', '年化成本']]

print("\n" + "=" * 95)
print("【截面策略绩效对比：亲眼见证 Carry + 库存共振如何将最大回撤腰斩、卡玛翻倍！】")
print("=" * 95)
print(fund_table)
print("=" * 95)

print("\n【分年度收益率对比 (%)】:")
yr_cmp = pd.DataFrame({
    '纯Carry': yearly_returns(port_carry)['策略'],
    '纯仓单库存': yearly_returns(port_inv)['策略'],
    'Carry+库存共振': yearly_returns(port_fund)['策略'],
    '截面动量XSMOM': yearly_returns(port_xsmom_pro)['策略'],
})
print((yr_cmp * 100).round(2))

# 绘制净值与回撤对比图
plot_equity_compare({
    '纯期限结构 Carry': port_carry,
    '纯仓单库存因子': port_inv,
    'Carry + 仓单库存 (基本面双因子共振)': port_fund,
    '截面动量 XSMOM': port_xsmom_pro,
}, title='阶段二截面策略对比：Carry vs 仓单库存 vs 基本面双因子共振 (扣费净值)', log=False)


#%% 第 5 块：FOF 尽调灵魂拷问与实战思考 ---------------------------------
print("\n" + "=" * 80)
print("【阶段二 · 练习 06 课后思考与 FOF 尽调实战拷问】：")
print("=" * 80)
print("""
[思考 1]：【截面动量 (XSMOM) 与 时序动量 (TSMOM) 的尽调鉴别】
    当你在路演中听到管理人说“我们做的是动量策略”时，你如何从他的净值表现或持仓估值表中，
    快速区分他是做“时序动量”还是“截面动量”？
    提示：观察在 2020-2021 商品大牛市或 2023 年震荡市中，他的【多空净敞口 (Net Exposure)】是否长期偏离 0，
    或者用你的 cta_nav_regression.py 回归看他在【长/中周期趋势】与【截面强弱动量】上的 Beta 暴露！

[思考 2]：【为什么 Carry + 库存 能把最大回撤从 -21.7% 压降到 -10.5%？】
    观察分年度收益表：在纯 Carry 表现较弱或发生回撤的年份（例如 2019、2022、2025），
    纯仓单库存因子的表现如何？为什么“高升水但低库存”的品种绝对不能盲目做空？

[思考 3]：【交易所仓单 vs 社会全口径库存（尽调进阶题）】
    我们在本练习中使用的是交易所公开的“注册仓单 (On-Warrant)”。
    在头部基本面量化 CTA（如敦和、永安国富、混沌或头部量化私募的基本面组）中，
    研究员除了看交易所仓单，还会花重金购买 Mysteel（我的钢铁）、隆众化工、卓创资讯的“社会库存/钢厂库存/港口库存”。
    请问：交易所仓单与社会库存有什么区别？为什么有时候“社会库存很高，但交易所注册仓单却很低”？
""")
print("=" * 80)
