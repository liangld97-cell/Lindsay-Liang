# -*- coding: utf-8 -*-
"""
向量化回测 + 绩效评价

约定（务必理解）：
    signal[t] = 只用 t 时刻（含）及之前的数据，在 t 这根 bar 收盘后算出的“目标仓位”
                取值一般为 1（满仓做多）/ -1（满仓做空）/ 0（空仓），也可以是 -1~1 之间的小数
    仓位单位   = 名义市值 / 账户权益，1 表示 1 倍名义杠杆
    执行延迟由框架处理，你写信号时【不要】自己 shift。

execution 参数：
    'next_open'（默认）t 收盘出信号，t+1 开盘成交 —— 比较接近实盘
    'close'            t 收盘出信号，t 收盘就成交 —— 理想化，会高估收益
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_chinese_font():
    """让 matplotlib 正常显示中文和负号。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# 回测
# ----------------------------------------------------------------------
def run_backtest(df, signal, cost_rate=0.0004, execution='next_open', roll_cost=True):
    """
    参数
    ----
   
    signal & position    : 名义市值/账户权益
    gross: 日收益率
    cost : 交易成本
    net : 日净收益率
    equity：动态净值
    返回
    ----
    DataFrame：signal, position(当根 bar 收盘时的持仓), gross_ret, cost, net_ret, equity, drawdown
    """
    sig = signal.reindex(df.index).astype(float).fillna(0.0)
    close, open_ = df['close'], df['open']

    if execution == 'close':
        pos = sig.shift(1).fillna(0.0)                 # bar t 期间持有的仓位
        gross = pos * close.pct_change().fillna(0.0)
        trade = sig.diff().abs().fillna(sig.abs())     # 在 t 收盘成交
        held_before_roll = pos
        position = sig                                 # t 收盘后的持仓
    elif execution == 'next_open':
        pos_old = sig.shift(2).fillna(0.0)             # bar t 开盘前持有
        pos_new = sig.shift(1).fillna(0.0)             # bar t 开盘成交后持有
        gap = (open_ / close.shift(1) - 1).fillna(0.0)
        intraday = (close / open_ - 1).fillna(0.0)
        gross = (1 + pos_old * gap) * (1 + pos_new * intraday) - 1
        trade = (pos_new - pos_old).abs()
        held_before_roll = pos_old
        position = pos_new
    else:
        raise ValueError("execution 只能是 'next_open' 或 'close'")

    cost = trade * cost_rate
    if roll_cost and 'roll' in df.columns:
        cost = cost + df['roll'].astype(float) * held_before_roll.abs() * 2 * cost_rate

    net = gross - cost
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1

    return pd.DataFrame({
        'signal': sig, 'position': position, 'gross_ret': gross,
        'cost': cost, 'net_ret': net, 'equity': equity, 'drawdown': drawdown,
    })


# ----------------------------------------------------------------------
# 交易明细与绩效
# ----------------------------------------------------------------------
def trade_list(result):
    """把连续同方向的持仓视为一笔交易（开仓成本计入本笔，平仓成本计入下一段）。"""
    sign = np.sign(result['position'])
    trade_id = (sign != sign.shift()).cumsum()
    rows = []
    for _, grp in result.groupby(trade_id):
        s = np.sign(grp['position'].iloc[0])
        if s == 0:
            continue
        rows.append({
            '开始': grp.index[0], '结束': grp.index[-1],
            '方向': '多' if s > 0 else '空',
            '持有bar数': len(grp),
            '收益': (1 + grp['net_ret']).prod() - 1,
        })
    return pd.DataFrame(rows)


def performance(result, periods_per_year=252):
    """计算常用绩效指标，返回一个 Series。"""
    r, eq, dd = result['net_ret'], result['equity'], result['drawdown']
    
    n = len(r)
    
    ann_ret = eq.iloc[-1] ** (periods_per_year / n) - 1
    ann_vol = r.std() * np.sqrt(periods_per_year) 
    sharpe = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    downside = np.sqrt((r.clip(upper=0) ** 2).mean())  # 只挑出亏损的日子，把赚钱的日子全部抹平为0
    sortino = r.mean() / downside * np.sqrt(periods_per_year) if downside > 0 else np.nan
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    underwater = dd < 0
    longest_dd = underwater.groupby((~underwater).cumsum()).sum().max()

    monthly = (1 + r).resample('M').prod() - 1

    out = {
        '年化收益': ann_ret,
        '年化波动': ann_vol,
        '夏普比率': sharpe,
        '索提诺比率': sortino,
        '最大回撤': mdd,
        '卡玛比率': calmar,
        '最长回撤期(bar)': longest_dd,
        '日收益偏度': r.skew(),
        '月收益偏度': monthly.skew(),
        '月度胜率': (monthly > 0).mean(),
    }

    if 'position' in result.columns:            # 单品种回测：附带逐笔交易统计
        trades = trade_list(result)
        if len(trades):
            wins = trades.loc[trades['收益'] > 0, '收益']
            losses = trades.loc[trades['收益'] <= 0, '收益']
            win_rate = len(wins) / len(trades)
            pl_ratio = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan
            avg_hold = trades['持有bar数'].mean()
        else:
            win_rate = pl_ratio = avg_hold = np.nan
        out.update({
            '交易次数': len(trades),
            '交易胜率': win_rate,
            '盈亏比': pl_ratio,
            '平均持仓(bar)': avg_hold,
            '年化换手(倍)': result['position'].diff().abs().sum() / n * periods_per_year,
            '持仓时间占比': (result['position'] != 0).mean(),
        })
    else:                                        # 组合回测（portfolio_backtest 的结果）
        out.update({
            '年化换手(倍)': result['turnover'].sum() / n * periods_per_year,
            '年化成本': result['cost'].sum() / n * periods_per_year,
            '平均名义杠杆': result['leverage'].mean(),
            '平均持仓品种数': result['n_assets'].mean(),
        })
    return pd.Series(out)


def yearly_returns(result, df=None):
    """分年度收益。传入 df 时附带买入持有作为对比。"""
    r = result['net_ret']
    out = pd.DataFrame({'策略': (1 + r).groupby(r.index.year).prod() - 1})
    if df is not None:
        bh = df['close'].pct_change().fillna(0)
        out['买入持有'] = (1 + bh).groupby(bh.index.year).prod() - 1
    out.index.name = '年份'
    return out


def plot_result(result, df=None, title=''):
    """三联图：净值 / 回撤 / 仓位。"""
    setup_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.2, 1]})
    axes[0].plot(result['equity'], label='策略净值', lw=1.3)
    if df is not None:
        axes[0].plot(df['close'] / df['close'].iloc[0], label='买入持有', lw=1, alpha=0.6)
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(result.index, result['drawdown'], 0, color='tab:red', alpha=0.4)
    axes[1].set_ylabel('回撤')
    axes[1].grid(alpha=0.3)

    axes[2].step(result.index, result['position'], where='post', lw=0.8)
    axes[2].set_ylabel('仓位')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 未来函数检查
# ----------------------------------------------------------------------
def check_lookahead(signal_func, df, n_checks=5, **kwargs):
    """
    检查信号函数是否偷看了未来数据。
    原理：如果信号只用历史数据，那么“用截断后的数据算出的信号”应该和
         “用全部数据算出的信号”在截断点之前完全一致。
    """
    full = signal_func(df, **kwargs)
    if not isinstance(full, pd.Series) or not full.index.equals(df.index):
        print('[警告] 信号必须是与 df.index 完全对齐的 pd.Series')
        return False
    cuts = np.linspace(len(df) * 0.3, len(df) - 1, n_checks).astype(int)
    for c in cuts:
        part = signal_func(df.iloc[:c], **kwargs)
        a = full.iloc[:c].astype(float).fillna(-999).values
        b = part.reindex(full.index[:c]).astype(float).fillna(-999).values
        if not np.allclose(a, b):
            bad = full.index[:c][~np.isclose(a, b)][0]
            print(f'[警告] 发现未来函数！截断到 {df.index[c - 1].date()} 时，'
                  f'{bad.date()} 的信号与全样本结果不一致')
            return False
    print('[通过] 未发现未来函数')
    return True
