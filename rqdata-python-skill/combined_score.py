import rqdatac
import pandas as pd
import numpy as np
rqdatac.init()
from rqdatac import get_factor, get_price

# ---------- 载入91只基本面数据 ----------
fund = pd.read_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/financial_details_91.csv")
codes = fund['代码'].tolist()

# ---------- 技术因子 ----------
tech = get_factor(codes, ['MA20', 'MA60', 'MA250', 'RSI6', 'RSI10', 'MACD_DIFF', 'MACD_DEA', 'MACD_HIST']).reset_index()
tech = tech[tech['date'] == tech['date'].max()].copy()
t = tech.set_index('order_book_id')

# ---------- 价格动量（前复权，近1年） ----------
px = get_price(codes, '20250801', '20260809', frequency='1d', fields=['close'], adjust_type='pre', expect_df=True)
px = px.reset_index()
px['date'] = pd.to_datetime(px['date'])
last = px.groupby('order_book_id')['close'].last()
ret_6m = last / px[px['date'] <= pd.Timestamp('2026-02-09')].groupby('order_book_id')['close'].last() - 1
ret_1y = last / px.groupby('order_book_id')['close'].first() - 1

# ---------- 基本面打分 ----------
f = fund.set_index('代码')
def score_f(row):
    s = 0
    roe = row['摊薄ROE(%)']; s += min(roe / 1.5, 20)
    inc = row['扣非净利同比(%)']; s += min(max(inc, -20), 40) * 0.5
    s += min(max(row['营收同比(%)'], -20), 30) * 0.3
    pe = row['市盈率']; s += max(20 - pe, 2) * 0.8 if pe > 0 else 0
    dy = row['股息率(%)']; s += min(dy * 1.5, 10)
    cf = row['盈余现金保障倍数']; s += min(max(cf, 0), 2) * 5
    if row['名称'] not in ['新华保险', '中国人寿', '中国太保']:
        debt = row['资产负债率(%)']; s += max(10 - max(debt - 40, 0) * 0.35, 0)
    else:
        s += 5
    return s

f['fund_score'] = f.apply(score_f, axis=1)

# ---------- 技术面打分 ----------
def score_t(code):
    s = 0
    try:
        r = t.loc[code]
        close = last.get(code, np.nan)
        ma60, ma250 = r['MA60'], r['MA250']
        if np.isnan(close) or np.isnan(ma60): return 0
        if close > ma60 and (np.isnan(ma250) or close > ma250): s += 30
        elif close > ma60: s += 15
        elif not np.isnan(ma250) and close > ma250: s += 10
        m6 = ret_6m.get(code, np.nan)
        if not np.isnan(m6):
            if 0 <= m6 <= 0.6: s += 30 * (m6 / 0.6)
            elif m6 < 0: s += 0
            else: s += 15  # 涨太多追高风险
        rsi = r['RSI6']
        if not np.isnan(rsi):
            if 40 <= rsi <= 65: s += 20
            elif 30 <= rsi < 40 or 65 < rsi <= 75: s += 12
            elif 20 <= rsi < 30 or 75 < rsi <= 85: s += 6
        diff, dea = r['MACD_DIFF'], r['MACD_DEA']
        if not np.isnan(diff) and not np.isnan(dea):
            if diff > dea: s += 20
    except Exception:
        pass
    return s

f['tech_score'] = [score_t(c) for c in f.index]

# ---------- 综合 ----------
f['combined'] = 0.6 * f['fund_score'] + 0.4 * f['tech_score']
f['6月涨幅(%)'] = (ret_6m.reindex(f.index) * 100).round(1)
f['1年涨幅(%)'] = (ret_1y.reindex(f.index) * 100).round(1)
f['RSI6'] = t['RSI6'].reindex(f.index).round(1)
f['站上MA60'] = (last.reindex(f.index) > t['MA60'].reindex(f.index)).map({True: '是', False: '否'})

out = f[['名称', '市盈率', '摊薄ROE(%)', '扣非净利同比(%)', '股息率(%)',
         'fund_score', 'tech_score', 'combined', '6月涨幅(%)', '1年涨幅(%)', 'RSI6', '站上MA60']]
out = out.sort_values('combined', ascending=False)
out[['fund_score', 'tech_score', 'combined']] = out[['fund_score', 'tech_score', 'combined']].round(1)

csv_path = r"C:/Users/Lenovo/Desktop/rqdata-python-skill/combined_ranking.csv"
out.to_csv(csv_path, encoding='utf-8-sig')
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', None)
print("综合排序 TOP25（基本面60% + 技术面40%）:")
print(out.head(25).to_string())
