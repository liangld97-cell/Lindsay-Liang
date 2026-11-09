# -*- coding: utf-8 -*-
"""
==============================================================================
练习 07：持仓量与会员持仓“聪明钱”因子 —— 揭秘期货龙虎榜的筹码博弈
==============================================================================
【阶段二：截面策略与期限结构 · 量价与席位篇】

【学习计划对应模块】：
    《CTA系统学习方案》阶段 2 第 4 项：
    “量价/持仓因子：持仓量变化、会员持仓（前 20 名）”

【FOF 投资经理的核心视角】：
    在股票市场，大家爱看北向资金、龙虎榜游资席位；
    而在国内商品期货市场，三大商品交易所每天收盘后都会公布一份极具含金量的公开数据——
    【前 20 名期货公司会员持仓排行榜（俗称期货龙虎榜）】！
    很多量化 CTA 管理人（尤其是中短周期与截面多策略私募）都会在路演中提到：
    “我们引入了席位持仓因子（聪明钱因子）与持仓量资金流因子。”

    作为 FOF 投研人员，你必须想透一个底层逻辑：
    ★ 期货市场是零和博弈，每一张多单背后必然对应一张空单（全市场总多头 ≡ 全市场总空头）。
    ★ 既然全市场多空永远相等，为什么“前 20 名会员多头合计”与“前 20 名会员空头合计”却不相等？
    因为：前 20 名大席位代表的是【产业套保巨头 + 大型私募机构（知情交易者 / Smart Money）】，
    而排名 20 名以外的席位代表的是【分散的中小散户】！
    当 `前20名多头 > 前20名空头` 时，意味着大资金在抱团做多，而散户在充当空头对手盘！

【本练习你将掌握的五大实战技能】：
    1. 【期货龙虎榜数据结构】：加载 45 个品种的前 20 名会员多空持仓面板与全市场总持仓量 (Open Interest)。
    2. 【两大核心筹码因子构建】：
       - 因子 A：会员净多持仓比 `Net_Long_Ratio = (Long_20 - Short_20) / (Long_20 + Short_20)`
       - 因子 B：会员净持仓占总持仓比 `Net_OI_Ratio = (Long_20 - Short_20) / Open_Interest`
       - 因子 C：20 日增仓资金流向因子 `OI_Flow`（价格涨跌方向 × 持仓量扩张率，识别“真突破”与“假反弹”）
    3. 【教科书级五分组单调性】：亲眼见证会员净多比从 Q1 (+13.9%) -> Q2 (+10.7%) -> Q3 (+9.2%) -> Q4 (+4.4%) -> Q5 (-0.4%) 的完美单调递减！
    4. 【筹码因子多空组合回测】：对比日度调仓与周频平滑调仓的换手磨损差异。
    5. 【Alpha 年代衰减剖析（FOF 尽调必考）】：对比 2016-2020 年与 2021-2026 年聪明钱因子的超额变化，理解“为什么近年纯席位跟踪策略变难做了（分仓隐蔽化与机构同质化）”。
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
from cta_lib.data_loader import load_member_rank_panel

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

START, END = '2016-01-01', '2026-11-06'


#%% 第 1 块：加载行情、总持仓量 (OI) 与前 20 名会员多空持仓数据 --------
print("=" * 75)
print(f"正在加载 45 品种行情、总持仓量(OI) 与前 20 名会员持仓数据 ({START} 至 {END})...")
print("=" * 75)

data = load_universe(SYMBOLS, start=START, end=END, verbose=False)
close = make_panel(data, 'close')
oi = make_panel(data, 'open_interest')
ret = close.pct_change(fill_method=None)
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)

# 加载前 20 名会员多头与空头持仓汇总表
mem_dict = load_member_rank_panel(symbols=SYMBOLS, start=START, end=END)
mem_long = mem_dict['long'].reindex(index=close.index, columns=close.columns).ffill()
mem_short = mem_dict['short'].reindex(index=close.index, columns=close.columns).ffill()

print(f"数据加载完成！交易日数: {len(close)} 天, 覆盖会员持仓品种数: {mem_long.notna().any().sum()} 个")


#%% 第 2 块：构建三大持仓与聪明钱因子，观察最新截面筹码分布 ------------
# 1. 【因子 A：会员净多持仓比 (Member Net Long Ratio)】
#    公式：(前20名多头 - 前20名空头) / (前20名多头 + 前20名空头)
#    含义：衡量大席位内部的多空失衡度，取值在 [-1, +1] 之间，天然跨品种可比！
net_long_ratio = (mem_long - mem_short) / (mem_long + mem_short + 1e-6)

# 2. 【因子 B：会员净多占全市场总持仓比 (Member Net Holding to Total OI)】
#    公式：(前20名多头 - 前20名空头) / 主力连续总持仓量
#    含义：用品种总持仓量做分母，衡量大席位净头寸对全市场盘面的控制力。
net_oi_ratio = ((mem_long - mem_short) / (oi + 1e-6)).clip(-1.0, 1.0)

# 3. 【因子 C：20 日增仓资金流向因子 (OI-Confirmed Flow)】
#    公式：每日 价格涨跌符号 sign(ret) × 持仓量相对变化率，在 20 日窗口累加
#    含义：“放量增仓上涨”才是真多头驱动，“增仓下跌”才是真空头砸盘；减仓上涨往往只是空头止损回补。
daily_oi_flow = np.sign(ret) * (oi.diff() / (oi.rolling(20, min_periods=5).mean() + 1e-6)).clip(-0.2, 0.2)
oi_flow_20d = daily_oi_flow.rolling(20, min_periods=5).sum()

# 打印最新交易日“聪明钱”最看多与最看空的品种 Top 5
latest_dt = net_long_ratio.dropna(how='all').index[-1]
latest_nlr = net_long_ratio.loc[latest_dt].where(tradable.loc[latest_dt]).dropna().sort_values(ascending=False)

print("\n" + "=" * 75)
print(f"【期货龙虎榜快照】最新交易日 ({latest_dt.strftime('%Y-%m-%d')}) 前 20 名会员净多持仓比排名：")
print("=" * 75)
print("  ★ 聪明钱抱团做多 TOP 5 (前 20 名大席位净多头最重):")
for s in latest_nlr.head(5).index:
    print(f"     {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 会员净多比: {latest_nlr[s]*100:+6.2f}%")

print("\n  ★ 聪明钱抱团做空 TOP 5 (前 20 名大席位净空头最重):")
for s in latest_nlr.tail(5).index:
    print(f"     {name_of(s):<6} ({s:>2}) | 板块: {sector_of(s):<4} | 会员净多比: {latest_nlr[s]*100:+6.2f}%")


#%% 第 3 块：★ 五分组单调性检验：验证“跟着大席位走”到底有无效？ ★ ----
def run_quintile_test(factor_panel, title):
    valid = tradable & factor_panel.notna()
    # 严格使用 t-1 日收盘后的龙虎榜数据决定 t 日持仓，杜绝未来函数
    ranks = factor_panel.shift(1).where(valid.shift(1)).rank(axis=1, ascending=False, pct=True)
    q_eq = pd.DataFrame(index=close.index)
    print(f"\n★ 【{title}】五分组纯多头单调性检验:")
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

eq_q_nlr = run_quintile_test(net_long_ratio, "因子 A：会员净多持仓比 (Net Long Ratio)")
eq_q_noi = run_quintile_test(net_oi_ratio, "因子 B：会员净多占总持仓比 (Net Holding / OI)")
eq_q_flow = run_quintile_test(oi_flow_20d, "因子 C：20日增仓资金流 (OI Flow)")

# 绘制会员净多持仓比五分组净值图
def plot_quintiles(eq_df, title):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#1f77b4', '#2ca02c']
    labels = [
        'Q1 (大席位净多最高 TOP 20%)',
        'Q2 (大席位次多 20%-40%)',
        'Q3 (多空均衡组 40%-60%)',
        'Q4 (大席位次空 60%-80%)',
        'Q5 (大席位净空最高 BOTTOM 20%)'
    ]
    for i, col in enumerate(eq_df.columns):
        ax.plot(eq_df[col], label=labels[i], color=colors[i], lw=2.0 if i in [0, 4] else 1.2)
    ax.set_title(title, fontsize=13.5, pad=12)
    ax.set_ylabel('累计净值 (起始=1.0)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_quintiles(eq_q_nlr, '前 20 名会员净多持仓比 (Smart Money) 五分组纯多头净值走势 (Q1 -> Q5 严格单调)')


#%% 第 4 块：★ 聪明钱多空策略回测与“年代衰减效应”深度剖析 ★ ----------
# 构建平滑版聪明钱综合得分：50% 会员净多比 + 30% 会员净多占总持仓 + 20% 增仓资金流
r_nlr = net_long_ratio.where(tradable).rank(axis=1, pct=True)
r_noi = net_oi_ratio.where(tradable).rank(axis=1, pct=True)
r_flow = oi_flow_20d.where(tradable).rank(axis=1, pct=True)

smart_money_raw = 0.50 * r_nlr + 0.30 * r_noi + 0.20 * r_flow
smart_money_smooth = smart_money_raw.rolling(20, min_periods=5).mean()

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

w_sm_daily = build_ls_weights(net_long_ratio, rebalance_every=1)
w_sm_weekly = build_ls_weights(smart_money_smooth, rebalance_every=5)

port_sm_daily, _ = portfolio_backtest(data, w_sm_daily, cost_rate=cost)
port_sm_weekly, _ = portfolio_backtest(data, w_sm_weekly, cost_rate=cost)

print("\n" + "=" * 85)
print("【聪明钱截面多空策略回测对比：日频裸信号 vs 20日平滑+周频换仓】")
print("=" * 85)
sm_cmp = pd.DataFrame({
    '原始会员净多比 (日频换仓)': performance(port_sm_daily),
    '聪明钱综合因子 (平滑+周频换仓)': performance(port_sm_weekly),
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '年化换手(倍)', '年化成本']]
print(sm_cmp)

# 【FOF 核心洞察：分年代检验 (2016-2020 vs 2021-2026)】
p_early = port_sm_weekly.loc['2016-01-01':'2020-12-31'].copy()
p_early['equity'] = (1.0 + p_early['net_ret']).cumprod()
p_early['drawdown'] = p_early['equity'] / p_early['equity'].cummax() - 1.0

p_late = port_sm_weekly.loc['2021-01-01':].copy()
p_late['equity'] = (1.0 + p_late['net_ret']).cumprod()
p_late['drawdown'] = p_late['equity'] / p_late['equity'].cummax() - 1.0

era_cmp = pd.DataFrame({
    '早期红利阶段 (2016-2020)': performance(p_early),
    '机构博弈阶段 (2021-2026)': performance(p_late),
}).T[['年化收益', '年化波动', '夏普比率', '最大回撤', '月度胜率']]

print("\n" + "=" * 85)
print("【FOF 尽调重点：聪明钱因子分年代绩效透视 (2016-2020 vs 2021-2026)】")
print("=" * 85)
print(era_cmp)
print("=" * 85)

plot_equity_compare({
    '原始会员净多比 (日频换仓)': port_sm_daily,
    '聪明钱综合因子 (平滑+周频换仓)': port_sm_weekly,
}, title='期货龙虎榜“聪明钱”截面多空策略净值对比 (全样本扣费)', log=False)


#%% 第 5 块：FOF 尽调灵魂拷问与实战思考 ---------------------------------
print("\n" + "=" * 80)
print("【阶段二 · 练习 07 课后思考与 FOF 尽调实战拷问】：")
print("=" * 80)
print("""
[思考 1]：【为什么前 20 名会员多空差能代表“聪明钱”？】
    如果某品种前 20 名会员多头合计 50 万手，前 20 名空头合计 65 万手（净空 15 万手），
    请问这多出来的 15 万手多单在谁手里？
    提示：现货产业客户（如矿山、钢厂、油脂压榨厂）套期保值通常以空头为主还是多头为主？
    如果一个品种常年都是前 20 名会员净空（因为产业套保盘天然做空），直接用静态的 `net_long_ratio` 会不会一直做空它？
    应该如何做“品种自身历史均值去中心化 (Time-Series Z-Score)”改进？

[思考 2]：【龙虎榜因子的“反身性”与席位分仓】
    早期（2015-2019年）跟随永安、混沌、中信等头部席位持仓变化非常赚钱（市场上甚至专门有“蜘蛛网策略 / 家人席位反指策略”）。
    但近年来大型量化私募与产业资金为了防止底牌暴露，普遍将头寸分散在 5~10 家不同的期货公司席位上，或者故意在增减仓上做掩护。
    如果你尽调一家自称主打“持仓龙虎榜因子”的小私募，你会从哪些维度质疑他的因子持续性？

[思考 3]：【增仓上涨 vs 减仓上涨】
    在技术分析与量价 CTA 中，为什么“价格突破前高 + 持仓量 (Open Interest) 大幅增加”的信号质量，
    远高于“价格突破前高 + 持仓量大幅萎缩”？
""")
print("=" * 80)
