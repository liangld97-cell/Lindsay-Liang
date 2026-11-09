# -*- coding: utf-8 -*-
"""
==============================================================================
CTA 市场环境气象雷达监控系统 (CTA Market Regime Radar)
==============================================================================
用途：
    每日收盘后或每周一晨会运行，量化评估当前商品期货市场对于 CTA 策略的宏观友好度。
    从三个公允、正交的维度（趋势质量、波动空间、板块共振）进行扫描，
    提供客观的“顺风/逆风”判定，指导 FOF 配置与投后预期管理。

指标体系：
    1. 趋势效率比 (KER, Kaufman Efficiency Ratio)：度量路径顺畅度与假突破频率。
    2. 市场年化波动率历史分位：度量市场振幅空间与策略利润燃料。
    3. 60日均线多头广度 (Market Breadth)：度量全市场 45 品种同向共振程度。
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# 自动定位项目根目录并导入 cta_lib
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cta_lib.universe import SYMBOLS, name_of, sector_of
from cta_lib.portfolio import load_universe, make_panel

def run_cta_radar(start_date='2020-01-01', verbose=False):
    """运行 CTA 气象雷达监控"""
    print("\n" + "=" * 65)
    print(" 📡 正在加载商品市场数据并扫描 CTA 宏观气象...")
    print("=" * 65)
    
    # 1. 加载 45 个商品期货主力连续行情
    data = load_universe(SYMBOLS, start=start_date, verbose=verbose)
    close = make_panel(data, 'close')
    last_date = close.index[-1].strftime('%Y-%m-%d')
    
    # =========================================================================
    # 维度一：趋势效率比 (KER 20日) —— 识别“假突破与折返跑”
    # =========================================================================
    net_change = (close - close.shift(20)).abs()
    total_path = close.diff().abs().rolling(20).sum()
    ker_panel = (net_change / total_path)
    current_ker_series = ker_panel.iloc[-1].dropna()
    mkt_ker = current_ker_series.median()
    
    # 判定
    if mkt_ker >= 0.35:
        ker_status = "☀️ [趋势极顺畅] 单边行情纯度高，假突破率低"
        ker_signal = "晴"
    elif mkt_ker >= 0.22:
        ker_status = "🌤️ [趋势较顺畅] 局部品种有持续性单边"
        ker_signal = "多云"
    else:
        ker_status = "⛈️ [拉锯折返严重] 假突破频发，震荡反复打脸风险高"
        ker_signal = "雷暴"

    # =========================================================================
    # 维度二：全市场波动率历史分位 —— 识别“死水期与利润空间”
    # =========================================================================
    ret = close.pct_change()
    vol_panel = ret.rolling(20).std() * np.sqrt(252)
    hist_mkt_vol = vol_panel.median(axis=1).dropna()
    current_vol = hist_mkt_vol.iloc[-1]
    vol_percentile = (hist_mkt_vol < current_vol).mean()
    
    # 判定
    if vol_percentile >= 0.70:
        vol_status = "🔥 [高波动亢奋] 利润空间充足，警惕极端反转风险"
        vol_signal = "高波"
    elif vol_percentile >= 0.25:
        vol_status = "✅ [波动适中健康] 处于正常博弈区间，能够覆盖交易磨损"
        vol_signal = "正常"
    else:
        vol_status = "❄️ [极度低波死寂] 市场严重缩量，换手及换月磨损压力大"
        vol_signal = "低波"

    # =========================================================================
    # 维度三：60日均线多头广度 (Market Breadth) —— 识别“板块共振还是撕裂”
    # =========================================================================
    ma_60 = close.rolling(60).mean()
    above_ma = (close > ma_60).iloc[-1].dropna()
    breadth = above_ma.mean()
    
    # 判定
    if breadth >= 0.75:
        breadth_status = "🚀 [全市场多头共振] 宏观大单边做多行情，Beta 红利丰厚"
        breadth_signal = "全多共振"
    elif breadth <= 0.25:
        breadth_status = "📉 [全市场空头共振] 宏观大单边做空行情，空头顺风期"
        breadth_signal = "全空共振"
    elif 0.40 <= breadth <= 0.60:
        breadth_status = "⚠️ [板块严重撕裂分化] 多空势均力敌互相对冲，缺乏主线"
        breadth_signal = "撕裂内耗"
    else:
        breadth_status = "🌤️ [结构性偏向] 结构分化，局部板块有独立行情"
        breadth_signal = "结构行情"

    # =========================================================================
    # 板块细分透视 (Sector Breakdown)
    # =========================================================================
    sector_data = []
    for s in above_ma.index:
        sec = sector_of(s)
        k_val = current_ker_series.get(s, np.nan)
        is_up = above_ma.get(s, False)
        # 近20日涨跌幅
        r20 = (close[s].iloc[-1] / close[s].iloc[-21] - 1.0) if len(close) > 20 else 0.0
        sector_data.append({'板块': sec, '品种': s, '多头': is_up, 'KER': k_val, '20日涨跌': r20})
    
    sec_df = pd.DataFrame(sector_data)
    sec_summary = sec_df.groupby('板块').agg(
        品种数量=('品种', 'count'),
        多头占比=('多头', lambda x: f"{x.mean()*100:.1f}%"),
        平均20日涨跌=('20日涨跌', lambda x: f"{x.mean()*100:+.2f}%"),
        平均KER=('KER', lambda x: f"{x.mean():.2f}")
    )

    # =========================================================================
    # 输出综合监控面板
    # =========================================================================
    print("\n" + "═" * 65)
    print(f"      CTA 市场环境气象雷达监控日评 (截至 {last_date})")
    print("═" * 65)
    print(f" 1. 趋势效率中位数 (KER 20D):  {mkt_ker:.2f}  ───> {ker_status}")
    print(f" 2. 全市场年化波动率:        {current_vol*100:.1f}% (历史分位 {vol_percentile*100:.1f}%) ───> {vol_status}")
    print(f" 3. 60日均线多头广度:        {breadth*100:.1f}% ───> {breadth_status}")
    print("─" * 65)
    
    print(" 📊 各板块微观强弱分布透视：")
    print(sec_summary.to_string())
    print("─" * 65)
    
    # 投资经理行动指南
    print(" 💡 FOF 投资经理配置与预期指引：")
    if breadth_signal in ["全多共振", "全空共振"] and mkt_ker >= 0.25:
        print("  【配置建议】大晴天顺风期：趋势跟踪 CTA 处于爆发红利期，组合收益斜率高，可放心超配。")
    elif breadth_signal == "撕裂内耗":
        print("  【配置建议】多云分化期：板块互相对冲，长周期单边趋势收益平淡；")
        print("             【首选配置】：重仓「中高频量化 CTA」或「产业基本面截面多空对冲」，")
        print("             利用板块间强弱分化吃截面超额，避免单押单边大趋势。")
    elif vol_signal == "低波" or ker_signal == "雷暴":
        print("  【配置建议】逆风受难期：假突破或低波磨损，趋势类产品普遍回撤；")
        print("             【投后沟通】：做好客户预期管理，提示此属正常周期磨损，严禁恐慌赎回；")
        print("             【左侧建仓】：若心仪投顾回撤逼近历史极值，反而是绝佳的逆向加仓窗口！")
    else:
        print("  【配置建议】常态平衡期：各类 CTA 均有发挥空间，建议按 5:5 均衡配置趋势与截面套利。")
    print("═" * 65 + "\n")

    # 联动输出自建六大 CTA 策略基准指数的每日收益监控看板
    try:
        from cta_strategy_indices import build_cta_style_indices, print_daily_strategy_monitor
        nav_df, ret_df, port_dict, _ = build_cta_style_indices(
            start_date=start_date, save_excel=True, verbose=False
        )
        print_daily_strategy_monitor(nav_df, ret_df, port_dict)
    except Exception as e:
        print(f" [提示] 策略指数看板加载跳过: {e}")


if __name__ == '__main__':
    run_cta_radar()
