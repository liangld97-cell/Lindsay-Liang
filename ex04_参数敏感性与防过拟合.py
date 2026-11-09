# -*- coding: utf-8 -*-
"""
==============================================================================
练习 04：参数敏感性与防过拟合 —— 撕下神仙回测的画皮
==============================================================================
【FOF 投资经理的核心痛点】：
    经常有私募管理人拿着年化 40%、夏普 3.0 的完美回测来路演，但一买进实盘就大幅回撤。
    为什么回测神勇，实盘拉胯？
    99% 的原因都死在：【参数过度优化（Overfitting）】与【数据窥探偏差（Data Snooping）】！

【本练习你将掌握的四大实战核武器】：
    1. 【参数网格搜索与二维热力图】：如何通过夏普矩阵，肉眼识别“参数平原”与“参数孤岛”？
    2. 【样本内(IS)与样本外(OOS)盲测】：验证过拟合冠军在未见过的真实行情里的“猝死定律”。
    3. 【全市场多品种平原效应】：为什么分散到 45 个品种能天然把孤岛抚平成辽阔平原？
    4. 【步进优化 (Walk-Forward Analysis, WFA)】：模拟基金经理真实的实盘滚动训练与拼接检验。
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
from cta_lib.data_loader import load_dominant
from cta_lib.backtest import run_backtest, performance, yearly_returns, setup_chinese_font
from cta_lib.portfolio import (load_universe, make_panel, tradable_mask, estimate_cost_rates,
                               vol_weights, scale_to_target_vol, portfolio_backtest,
                               plot_equity_compare)

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

START, END = '2010-01-04', '2026-10-30'
IS_SPLIT_DATE = '2020-12-31'  # 样本内 (IS: 2010~2020) 与 样本外 (OOS: 2021~2026) 分界线


def plot_param_heatmap(matrix, title='参数敏感性夏普热力图', xlabel='慢线周期 (Slow)', ylabel='快线周期 (Fast)', cmap='RdYlGn'):
    """绘制高颜值参数网格热力图（无第三方库依赖）"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix.values, cmap=cmap, aspect='auto')
    
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    
    v_min, v_max = matrix.values.min(), matrix.values.max()
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                # 动态自适应文字颜色
                ratio = (val - v_min) / (v_max - v_min + 1e-6)
                text_color = "black" if (0.3 < ratio < 0.7) else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
                
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


#%% 第 1 块：数据加载与单品种/全市场准备 ---------------------------------
print("正在加载数据...")
# 1. 单品种：螺纹钢 RB（用于细致解剖参数敏感性）
df_rb = load_dominant('RB', start=START, end=END)

# 2. 全市场：45 个品种（用于多品种平原效应检验）
data = load_universe(SYMBOLS, start=START, end=END, verbose=False)
close = make_panel(data, 'close')
ret = close.pct_change()
tradable = tradable_mask(data).reindex(close.index).fillna(False)
cost = estimate_cost_rates(data)
print(f"数据加载完成！全样本区间: {START} 至 {END}")


#%% 第 2 块：★ 实验 A：单品种网格搜索与“参数孤岛”原形毕露 ★ ----------
# 设定快慢均线参数网格
fast_list = [5, 10, 15, 20, 25, 30, 40]
slow_list = [30, 40, 50, 60, 70, 80, 100, 120]

def sig_ma_fast(close_series, fast, slow):
    ma_f = close_series.rolling(fast).mean()
    ma_s = close_series.rolling(slow).mean()
    sig = pd.Series(0.0, index=close_series.index)
    sig[ma_f > ma_s] = 1.0
    sig[ma_f < ma_s] = -1.0
    return sig

print("\n正在对螺纹钢 RB 进行全样本网格搜索 (56 组参数)...")
rb_sharpe_grid = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)
rb_calmar_grid = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)

for f in fast_list:
    for s in slow_list:
        if f >= s:
            rb_sharpe_grid.loc[f, s] = np.nan
            continue
        sig = sig_ma_fast(df_rb['close'], f, s)
        res = run_backtest(df_rb, sig, cost_rate=0.0002)
        rb_sharpe_grid.loc[f, s] = performance(res).loc['夏普比率']
        rb_calmar_grid.loc[f, s] = performance(res).loc['卡玛比率']

print("螺纹钢 (RB) 单品种不同快慢均线夏普比率矩阵：")
print(rb_sharpe_grid.round(2))

# 画出单品种热力图
plot_param_heatmap(rb_sharpe_grid, title='螺纹钢单品种双均线夏普比率热力图 (全样本)')


#%% 第 3 块：★ 实验 B：样本内(IS)过度优化 vs 样本外(OOS)打折定律 ★ -----
# 划分样本内（2010~2020 优化参数）与 样本外（2021~2026 盲测）
df_rb_is = df_rb.loc[:IS_SPLIT_DATE]
df_rb_oos = df_rb.loc[IS_SPLIT_DATE:]

print(f"\n正在进行【样本内 (2010~2020)】参数网格训练...")
is_sharpe_grid = pd.DataFrame(index=fast_list, columns=slow_list, dtype=float)

for f in fast_list:
    for s in slow_list:
        if f >= s:
            continue
        sig = sig_ma_fast(df_rb_is['close'], f, s)
        res = run_backtest(df_rb_is, sig, cost_rate=0.0002)
        is_sharpe_grid.loc[f, s] = performance(res).loc['夏普比率']

# 找出样本内夏普最高的【孤岛冠军】
best_fast = is_sharpe_grid.stack().idxmax()[0]
best_slow = is_sharpe_grid.stack().idxmax()[1]
best_is_sharpe = is_sharpe_grid.loc[best_fast, best_slow]

# 选取工业界经典的【参数平原代表】(20, 60)
robust_fast, robust_slow = 20, 60
robust_is_sharpe = is_sharpe_grid.loc[robust_fast, robust_slow]

print(f"👉 样本内最优冠军组: ({best_fast}, {best_slow})，样本内夏普 = {best_is_sharpe:.3f}")
print(f"👉 经典平原基准组:   ({robust_fast}, {robust_slow})，样本内夏普 = {robust_is_sharpe:.3f}")

# 拿去样本外 (2021~2026) 接受真实历史盲测！
sig_best_oos = sig_ma_fast(df_rb_oos['close'], best_fast, best_slow)
res_best_oos = run_backtest(df_rb_oos, sig_best_oos, cost_rate=0.0002)

sig_robust_oos = sig_ma_fast(df_rb_oos['close'], robust_fast, robust_slow)
res_robust_oos = run_backtest(df_rb_oos, sig_robust_oos, cost_rate=0.0002)

compare_oos = pd.DataFrame({
    f'样本内最优({best_fast},{best_slow})': [
        best_is_sharpe,
        performance(res_best_oos).loc['夏普比率'],
        performance(res_best_oos).loc['年化收益'],
        performance(res_best_oos).loc['最大回撤']
    ],
    f'平原基准组({robust_fast},{robust_slow})': [
        robust_is_sharpe,
        performance(res_robust_oos).loc['夏普比率'],
        performance(res_robust_oos).loc['年化收益'],
        performance(res_robust_oos).loc['最大回撤']
    ]
}, index=['样本内夏普(IS)', '样本外夏普(OOS)', '样本外年化收益', '样本外最大回撤'])

print("\n===== 样本外盲测结果对比（观察性能衰减程度）=====")
print(compare_oos)
plot_equity_compare({
    f'样本内冠军({best_fast},{best_slow})': res_best_oos,
    f'平原基准({robust_fast},{robust_slow})': res_robust_oos
}, title='样本外 (2021-2026) 净值对比：孤岛冠军 vs 平原基准', log=False)


#%% 第 4 块：★ 实验 C：全市场 45 品种组合的“参数平原抚平效应” ★ ------
# 检验问题：多品种组合是否会让参数敏感性大幅下降？
print("\n正在计算全市场 45 品种组合的不同参数夏普热力图（请稍候十几秒）...")
port_fast_list = [10, 15, 20, 30]
port_slow_list = [40, 60, 80, 100]

port_sharpe_grid = pd.DataFrame(index=port_fast_list, columns=port_slow_list, dtype=float)

for f in port_fast_list:
    for s in port_slow_list:
        p_sig = pd.DataFrame({sym: sig_ma_fast(data[sym]['close'], f, s) for sym in SYMBOLS})
        p_sig_clean = p_sig.reindex(close.index).fillna(0.0).where(tradable, 0.0)
        w = scale_to_target_vol(vol_weights(p_sig_clean, ret), ret, target_vol=0.10)
        port, _ = portfolio_backtest(data, w, cost_rate=cost)
        port_sharpe_grid.loc[f, s] = performance(port).loc['夏普比率']

print("45 个商品多品种组合的参数夏普矩阵：")
print(port_sharpe_grid.round(2))
plot_param_heatmap(port_sharpe_grid, title='45 品种组合参数夏普热力图 (观察辽阔平原效应)')


#%% 第 5 块：★ 实验 D：步进优化 (Walk-Forward Analysis, WFA) ★ ---------
# 模拟真正的基金经理：每隔 1 年滚动用过去 3 年更新一次最优参数
print("\n正在执行 Walk-Forward 滚动步进模拟 (2014~2026 年)...")

TRAIN_YEARS = 3  # 滚动训练窗口：3 年
years = sorted(list(set(df_rb.index.year)))
wfa_start_year = years[0] + TRAIN_YEARS

wfa_signals = []
selected_params_record = []

for yr in range(wfa_start_year, years[-1] + 1):
    train_start = f"{yr - TRAIN_YEARS}-01-01"
    train_end = f"{yr - 1}-12-31"
    test_start = f"{yr}-01-01"
    test_end = f"{yr}-12-31"
    
    # 1. 在训练集上寻找样本内最优快慢均线参数
    df_train = df_rb.loc[train_start:train_end]
    best_f, best_s, max_sr = 20, 60, -999.0
    for f in [10, 20, 30]:
        for s in [40, 60, 80]:
            sig_tr = sig_ma_fast(df_train['close'], f, s)
            res_tr = run_backtest(df_train, sig_tr, cost_rate=0.0002)
            sr = performance(res_tr).loc['夏普比率']
            if sr > max_sr:
                max_sr = sr
                best_f, best_s = f, s
                
    selected_params_record.append({'年份': yr, '选中快线': best_f, '选中慢线': best_s, '训练期夏普': max_sr})
    
    # 2. 将最优参数应用到测试期盲跑（完全无未来函数）
    df_test = df_rb.loc[test_start:test_end]
    if len(df_test) > 0:
        sig_te = sig_ma_fast(df_test['close'], best_f, best_s)
        wfa_signals.append(sig_te)

wfa_sig_full = pd.concat(wfa_signals).sort_index()
df_rb_wfa = df_rb.loc[wfa_sig_full.index]
res_wfa = run_backtest(df_rb_wfa, wfa_sig_full, cost_rate=0.0002)

# 与固定参数 (20, 60) 在同一时期对比
sig_fixed = sig_ma_fast(df_rb_wfa['close'], 20, 60)
res_fixed = run_backtest(df_rb_wfa, sig_fixed, cost_rate=0.0002)

print("\n历年 Walk-Forward 动态选出的参数：")
print(pd.DataFrame(selected_params_record).to_string(index=False))

print("\n===== Walk-Forward 动态步进 vs 固定参数 绩效对比 =====")
perf_wfa_compare = pd.DataFrame({
    'Walk-Forward 步进优化': performance(res_wfa),
    '静态固定参数 (20, 60)': performance(res_fixed)
})
print(perf_wfa_compare.loc[['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率']])

plot_equity_compare({
    'Walk-Forward 动态步进': res_wfa,
    '固定参数 (20, 60)': res_fixed
}, title='Walk-Forward 动态步进优化 vs 静态固定参数 (螺纹钢实测)', log=False)


#%% 思考题（写在下面的注释里）------------------------------------------
# Q1.【参数孤岛辨识】观察螺纹钢的热力图，为什么某些孤立点夏普特别高，但周围全是绿变红？
#     如果你在尽调材料里看到投顾选了一组很“奇怪”的参数（如 13日均线配 47日均线），你会怎么想？
#
# Q2.【样本外衰减】实验 B 中，样本内挑选出的最优“冠军参数”，在样本外的表现有没有大幅打折？
#     为什么平原基准组反而在样本外更稳健？
#
# Q3.【分散抚平效应】对比实验 A（单品种热力图）和实验 C（45 品种组合热力图）：
#     为什么组合的热力图颜色更均匀、数值波动更小？这对降低策略研发中的过拟合风险有什么启示？
#
# Q4.【Walk-Forward 的真相】实验 D 中，每年滚动优化选出的参数在不断变化，
#     最终的 WFA 净值有没有显著战胜简单的“静态固定参数 (20, 60)”？
#     为什么频繁优化参数反而可能跑不过老老实实呆在参数平原中心的静态参数？
#
# Q5.【尽调夺命连环问】如果一个量化 CTA 投顾宣称他们模型每两周都会“用 AI / 遗传算法自动重新优化参数以适应市场最新变化”，
#     结合今天的 Walk-Forward 实验，你会怎么评估这种做法的实盘隐患？
