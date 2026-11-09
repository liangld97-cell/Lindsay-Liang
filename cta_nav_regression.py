# -*- coding: utf-8 -*-
"""
==============================================================================
程序二：外部私募 CTA 产品净值深度评估与风格暴露回归 (cta_nav_regression.py)
==============================================================================
功能：
    1. 读取固定格式的子基金净值表（第 1 列：日期，第 2 列：净值）。
    2. 全面计算产品自身风险收益特征：
       年化收益、年化波动率、下行波动率、夏普、索提诺、卡玛、偏度、峰度、胜率、
       最大回撤、最大回撤发生区间、最大回撤修复期（恢复创新高所需时间）。
    3. 与自建 1.0x 固定名义杠杆 CTA 基准指数做多元回归，同时输出：
       【名义资金暴露 Beta_1x】与【10%等波动率标准化暴露 Beta_10%Vol】。
==============================================================================
"""

#%% ★ 第 0 块：配置待评估的产品净值文件路径 ★ ---------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.backtest import setup_chinese_font
from cta_strategy_indices import build_cta_style_indices

pd.set_option('display.width', 240)
pd.set_option('display.max_rows', 100)
pd.set_option('display.unicode.east_asian_width', True)

# 在此填入待评估产品
FUND_FILE_PATH = r"D:\待评估产品净值\远澜火松.xlsx"
STYLE_COLS = ['长周期趋势', '中周期趋势', '短周期趋势', '截面强弱动量', '期限结构Carry', '商品多头Beta']


#%% 第 1 块：读取数据与计算产品自身详细绩效指标 -------------------------------
def load_fund_nav(file_path):
    """读取固定格式净值文件：第 1 列为日期，第 2 列为产品净值"""
    df = pd.read_csv(file_path) if file_path.lower().endswith('.csv') else pd.read_excel(file_path)
    dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    navs = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    s = pd.Series(navs.values, index=dates).dropna().sort_index()
    s = s[~s.index.duplicated(keep='last')]
    
    # s = s.loc['2025-03-28':]
    
    fund_name = os.path.splitext(os.path.basename(file_path))[0]
    return s, fund_name


def load_benchmark_nav(csv_path=None):
    """加载本地自建 1.0x 基准指数净值表（若不存在则自动生成）"""
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, 'CTA策略基准指数_日度净值.csv')
    if os.path.exists(csv_path):
        nav_df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        if all(c in nav_df.columns for c in STYLE_COLS):
            return nav_df
    nav_df, _, _, _ = build_cta_style_indices(start_date='2020-01-01', save_excel=True, verbose=True)
    return nav_df


def calc_fund_detailed_stats(nav_series, periods_per_year):
    """
    计算产品自身的详细收益、波动、偏度、峰度、最大回撤及回撤修复期
    """
    nav = nav_series / nav_series.iloc[0]
    ret = nav.pct_change().dropna()
    n = len(ret)

    # 1. 收益与波动率（总波动率 & 下行波动率）
    total_ret = nav.iloc[-1] - 1.0
    ann_ret = nav.iloc[-1] ** (periods_per_year / n) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(periods_per_year)

    downside_ret = np.minimum(ret, 0.0)
    downside_vol = np.sqrt(np.mean(downside_ret ** 2)) * np.sqrt(periods_per_year)

    # 2. 风险调整比率
    sharpe = (ann_ret-0.017) / (ann_vol + 1e-12)
    sortino = ann_ret / (downside_vol + 1e-12)

    # 3. 分布形态：偏度、峰度（超额峰度）、胜率
    skewness = ret.skew()
    kurtosis = ret.kurt()   # pandas 默认计算 Fisher 超额峰度（正态分布=0，>0 代表肥尾）
    win_rate = (ret > 0).mean()

    # 4. 最大回撤、回撤区间与回撤修复期
    cum_max = nav.cummax()
    dd = nav / cum_max - 1.0
    max_dd = dd.min()
    trough_date = dd.idxmin()
    peak_date = nav.loc[:trough_date].idxmax()
    peak_val = nav.loc[peak_date]

    # 查找从谷底之后首次重新突破前高 peak_val 的日期
    after_trough = nav.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_val]
    if len(recovered) > 0:
        recover_date = recovered.index[0]
        recover_days = (recover_date - trough_date).days
        recover_bars = len(nav.loc[trough_date:recover_date]) - 1
        recover_str = f"{recover_days} 自然日 ({recover_bars} 个净值期，于 {recover_date.strftime('%Y-%m-%d')} 修复)"
    else:
        recover_str = f"尚未修复 (自谷底已历时 {(nav.index[-1] - trough_date).days} 自然日)"

    calmar = ann_ret / (abs(max_dd) + 1e-12)

    return {
        '累计收益': total_ret,
        '年化收益': ann_ret,
        '年化波动率': ann_vol,
        '下行波动率': downside_vol,
        '夏普比率': sharpe,
        '索提诺比率': sortino,
        '卡玛比率': calmar,
        '最大回撤': max_dd,
        '最大回撤区间': f"{peak_date.strftime('%Y-%m-%d')} 至 {trough_date.strftime('%Y-%m-%d')}",
        '回撤修复期': recover_str,
        '收益偏度': skewness,
        '收益峰度': kurtosis,
        '单期胜率': win_rate,
    }


#%% 第 2 块：执行净值评估与多元风格回归 ---------------------------------------
def analyze_and_regress_fund(file_path, target_std_vol=0.10, plot=True):
    """一键读取子基金净值文件，输出详细绩效画像 + 六大CTA风格双口径回归报告"""
    s_fund, fund_name = load_fund_nav(file_path)
    index_nav_df = load_benchmark_nav()

    # 对齐基准指数区间内的有效日期
    valid_dates = s_fund.index[(s_fund.index >= index_nav_df.index.min()) &
                               (s_fund.index <= index_nav_df.index.max())]
    s_fund = s_fund.loc[valid_dates]
    matched_idx_nav = index_nav_df[STYLE_COLS].reindex(valid_dates, method='ffill')

    # 计算同区间收益率并识别频率
    y_ret = s_fund.pct_change().dropna()
    X_ret_nom = matched_idx_nav.pct_change().dropna()
    common_idx = y_ret.index.intersection(X_ret_nom.index)

    y = y_ret.loc[common_idx].values
    X_nom = X_ret_nom.loc[common_idx, STYLE_COLS].values
    n, k = X_nom.shape

    avg_gap_days = (common_idx[-1] - common_idx[0]).days / max(n, 1)
    periods_per_year = 252.0 if avg_gap_days <= 2.2 else (52.0 if avg_gap_days <= 9.0 else 12.0)
    freq_label = "日频" if periods_per_year == 252.0 else ("周频" if periods_per_year == 52.0 else "月频")

    # 1. 计算产品自身详细风险收益指标
    f_stats = calc_fund_detailed_stats(s_fund.loc[common_idx[0]:common_idx[-1]], periods_per_year)

    # 2. 多元线性回归与 10% 等波动率标准化转换
    idx_natural_vol = np.std(X_nom, axis=0, ddof=1) * np.sqrt(periods_per_year)
   
    X_design = np.column_stack([np.ones(n), X_nom])
    params_nom, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    alpha_per_period = params_nom[0]
    betas_nominal = params_nom[1:]
    betas_std10 = betas_std10 = betas_nominal * (idx_natural_vol / 0.10)

    y_hat = X_design @ params_nom
    resid = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / max(n - k - 1, 1)

    sigma2 = ss_res / max(n - k - 1, 1)
    cov_nom = sigma2 * np.linalg.pinv(X_design.T @ X_design)
    t_stats = params_nom / np.sqrt(np.maximum(np.diag(cov_nom), 1e-18))

    ann_alpha = alpha_per_period * periods_per_year
    ann_resid_vol = np.std(resid, ddof=1) * np.sqrt(periods_per_year)
    info_ratio = ann_alpha / (ann_resid_vol + 1e-8)

    risk_budget_share = np.abs(betas_std10) / (np.sum(np.abs(betas_std10)) + 1e-12)

    reg_table = pd.DataFrame({
        '指数自然波动': idx_natural_vol,
        '名义暴露(Beta_1x)': betas_nominal,
        '等波暴露(Beta_10%Vol)': betas_std10,
        '相对风险占比': risk_budget_share,
        't统计量': t_stats[1:],
        '显著性诊断': [
            '★★★ 极显著核心暴露' if abs(t) >= 2.58 else
            ('★★ 显著暴露' if abs(t) >= 1.96 else
             ('★ 弱暴露' if abs(t) >= 1.28 else '— 不显著'))
            for t in t_stats[1:]
        ]
    }, index=STYLE_COLS)

    # 3. 打印报告
    print("\n" + "═" * 102)
    print(f" 🔍 外部私募产品深度绩效画像与净值风格回归报告 —— 【{fund_name}】")
    print("═" * 102)
    print(f" 【一、产品自身风险收益与分布特征】 ({common_idx[0].strftime('%Y-%m-%d')} 至 {common_idx[-1].strftime('%Y-%m-%d')} | 共 {n} 个{freq_label}观测点)")
    print(f"  ▪ 收益与波动 : 累计收益 = {f_stats['累计收益']*100:+.2f}% | 年化收益 = {f_stats['年化收益']*100:+.2f}% | "
          f"年化波动率 = {f_stats['年化波动率']*100:.2f}% | 下行波动率 = {f_stats['下行波动率']*100:.2f}%")
    print(f"  ▪ 风险调整比 : 夏普比率 = {f_stats['夏普比率']:.2f} | 索提诺比率 = {f_stats['索提诺比率']:.2f} | "
          f"卡玛比率 = {f_stats['卡玛比率']:.2f} | {freq_label}胜率 = {f_stats['单期胜率']*100:.1f}%")
    print(f"  ▪ 高阶矩特征 : 收益偏度 (Skew) = {f_stats['收益偏度']:+.2f} | 超额峰度 (Kurt) = {f_stats['收益峰度']:+.2f} "
          f"({'正偏肥尾，具备爆发力' if f_stats['收益偏度'] > 0 else '负偏分布，需防尾部回撤'})")
    print(f"  ▪ 最大回撤   : {f_stats['最大回撤']*100:.2f}% (发生于 {f_stats['最大回撤区间']})")
    print(f"  ▪ 回撤修复期 : {f_stats['回撤修复期']}")
    print("─" * 102)
    print(" 【二、多元回归模型拟合与纯 Alpha 剥离】")
    print(f"  ▪ 模型解释度 : R² = {r_squared*100:.1f}% (调整后 R² = {adj_r_squared*100:.1f}%) "
          f"───> 六大公开CTA风格可解释该产品 {r_squared*100:.1f}% 的净值方差")
    print(f"  ▪ 剥离后Alpha: 年化纯 Alpha = {ann_alpha*100:+.2f}% (t = {t_stats[0]:+.2f}) | "
          f"特质波动率 = {ann_resid_vol*100:.2f}% | 信息比率 IR = {info_ratio:+.2f}")
    print("─" * 102)
    print(" 【三、六大 CTA 风格因子双口径暴露明细】")

    disp_df = pd.DataFrame(index=reg_table.index)
    disp_df['指数自然波动'] = reg_table['指数自然波动'].map(lambda x: f"{x*100:5.2f}%")
    disp_df['名义暴露(Beta_1x)'] = reg_table['名义暴露(Beta_1x)'].map(lambda x: f"{x:+.3f}")
    disp_df['等波暴露(Beta_10%Vol)'] = reg_table['等波暴露(Beta_10%Vol)'].map(lambda x: f"{x:+.3f}")
    disp_df['相对风险占比'] = reg_table['相对风险占比'].map(lambda x: f"{x*100:5.1f}%")
    disp_df['t 统计量'] = reg_table['t统计量'].map(lambda x: f"{x:+.2f}")
    disp_df['显著性诊断'] = reg_table['显著性诊断']
    print(disp_df.to_string())
    print("═" * 102 + "\n")

    # 4. 绘图
    if plot:
        setup_chinese_font()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={'width_ratios': [1.55, 1.05]})

        eq_actual = (1.0 + pd.Series(y, index=common_idx)).cumprod()
        eq_style_fit = (1.0 + pd.Series(X_nom @ betas_nominal, index=common_idx)).cumprod()
        eq_alpha = (1.0 + pd.Series(alpha_per_period + resid, index=common_idx)).cumprod()

        ax1.plot(eq_actual, label=f'{fund_name} (实际净值)', color='#1f77b4', lw=2.0)
        ax1.plot(eq_style_fit, label='六大CTA风格复制净值 (Beta拟合)', color='#ff7f0e', lw=1.6, linestyle='--')
        ax1.plot(eq_alpha, label=f'剥离风格后残差 Alpha (年化{ann_alpha*100:+.1f}%)', color='#2ca02c', lw=1.4)
        ax1.set_title(f'【{fund_name}】净值拆解 (R2={r_squared*100:.1f}%, 最大回撤={f_stats["最大回撤"]*100:.1f}%)', fontsize=11.5)
        ax1.set_ylabel('累计净值 (起点=1.0)')
        ax1.legend(loc='upper left', fontsize=9.5)
        ax1.grid(alpha=0.3)

        y_pos = np.arange(len(STYLE_COLS))
        bar_h = 0.35
        b_nom_rev = betas_nominal[::-1]
        b_std_rev = betas_std10[::-1]
        t_rev = t_stats[1:][::-1]

        ax2.barh(y_pos + bar_h/2, b_nom_rev, height=bar_h, label='名义资金暴露 (Beta_1x)', color='#1f77b4', alpha=0.85)
        ax2.barh(y_pos - bar_h/2, b_std_rev, height=bar_h, label='10%等波风险暴露 (Beta_10%Vol)', color='#d62728', alpha=0.85)
        ax2.axvline(0, color='black', lw=0.9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(STYLE_COLS[::-1], fontsize=10)

        for i, (bn, bs, tv) in enumerate(zip(b_nom_rev, b_std_rev, t_rev)):
            max_w = max(bn, bs) if max(bn, bs) >= 0 else min(bn, bs)
            offset = 0.02 if max_w >= 0 else -0.02
            ax2.text(max_w + offset, y_pos[i], f"等波:{bs:+.2f} (t={tv:+.1f})",
                     va='center', ha='left' if max_w >= 0 else 'right', fontsize=8.5)

        ax2.set_xlim(min(np.min(b_nom_rev), np.min(b_std_rev), -0.2) - 0.25,
                     max(np.max(b_nom_rev), np.max(b_std_rev), 0.5) + 0.40)
        ax2.set_title(f'【{fund_name}】双口径风格暴露对比', fontsize=11.5)
        ax2.set_xlabel('回归暴露系数 Beta')
        ax2.legend(loc='lower right', fontsize=8.5)
        ax2.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()

    return f_stats, reg_table


#%% 第 3 块：运行入口 ---------------------------------------------------------
if __name__ == '__main__':
    fund_stats, reg_result = analyze_and_regress_fund(FUND_FILE_PATH, plot=True)
