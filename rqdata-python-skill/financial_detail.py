import rqdatac
import pandas as pd
rqdatac.init()
from rqdatac import get_factor

# 读取上一步筛出的91只
src = pd.read_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/value_stocks_refined.csv")
codes = src['代码'].tolist()
name_map = dict(zip(src['代码'], src['名称']))

factors = [
    'pe_ratio_ttm', 'return_on_equity_diluted_ttm',
    'gross_profit_margin_ttm', 'net_profit_margin_ttm',
    'inc_revenue_ttm', 'inc_adjusted_net_profit_ttm',
    'debt_to_asset_ratio_ttm', 'operating_cash_flow_per_share_ttm',
    'surplus_cash_protection_multiples_ttm', 'dividend_yield_ttm',
    'market_cap_3',
]
raw = get_factor(codes, factors).reset_index()
raw = raw[raw['date'] == raw['date'].max()].copy()

out = pd.DataFrame({'代码': codes, '名称': codes})
out['名称'] = out['代码'].map(name_map)
for f in factors:
    out[f] = raw.set_index('order_book_id')[f].reindex(codes).values

# 转成易读单位
fmt = {
    'pe_ratio_ttm': '市盈率', 'return_on_equity_diluted_ttm': '摊薄ROE(%)',
    'gross_profit_margin_ttm': '毛利率(%)', 'net_profit_margin_ttm': '净利率(%)',
    'inc_revenue_ttm': '营收同比(%)', 'inc_adjusted_net_profit_ttm': '扣非净利同比(%)',
    'debt_to_asset_ratio_ttm': '资产负债率(%)', 'operating_cash_flow_per_share_ttm': '每股经营现金流(元)',
    'surplus_cash_protection_multiples_ttm': '盈余现金保障倍数', 'dividend_yield_ttm': '股息率(%)',
    'market_cap_3': '总市值(亿)',
}
out = out.rename(columns=fmt)
for c in ['摊薄ROE(%)', '毛利率(%)', '净利率(%)', '营收同比(%)', '扣非净利同比(%)', '资产负债率(%)', '股息率(%)']:
    out[c] = (out[c] * 100).round(2)
out['每股经营现金流(元)'] = out['每股经营现金流(元)'].round(2)
out['盈余现金保障倍数'] = out['盈余现金保障倍数'].round(2)
out['市盈率'] = out['市盈率'].round(2)
out['总市值(亿)'] = (out['总市值(亿)'] / 1e8).round(0)

cols = ['代码', '名称', '市盈率', '摊薄ROE(%)', '毛利率(%)', '净利率(%)',
        '营收同比(%)', '扣非净利同比(%)', '资产负债率(%)', '每股经营现金流(元)',
        '盈余现金保障倍数', '股息率(%)', '总市值(亿)']
out = out[cols]

csv_path = r"C:/Users/Lenovo/Desktop/rqdata-python-skill/financial_details_91.csv"
out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"共 {len(out)} 只，已保存到 {csv_path}")

# 关键警示：扣非净利负增长 / 高负债 / 现金流弱 / 股息率
warn_rev = out[out['扣非净利同比(%)'] < 0].sort_values('扣非净利同比(%)')
warn_debt = out[out['资产负债率(%)'] > 65].sort_values('资产负债率(%)', ascending=False)
warn_cash = out[out['盈余现金保障倍数'] < 0.8].sort_values('盈余现金保障倍数')
print(f"\n[警示1] 扣非净利负增长: {len(warn_rev)} 只")
print(warn_rev[['代码', '名称', '扣非净利同比(%)', '营收同比(%)']].to_string(index=False))
print(f"\n[警示2] 资产负债率>65%: {len(warn_debt)} 只")
print(warn_debt[['代码', '名称', '资产负债率(%)']].to_string(index=False))
print(f"\n[警示3] 盈余现金保障倍数<0.8(利润含金量低): {len(warn_cash)} 只")
print(warn_cash[['代码', '名称', '盈余现金保障倍数', '每股经营现金流(元)']].to_string(index=False))

# 亮点：高ROE+正增长+低负债+有股息
bright = out[(out['摊薄ROE(%)'] > 20) & (out['扣非净利同比(%)'] > 0) &
             (out['资产负债率(%)'] < 60) & (out['股息率(%)'] > 0)]
print(f"\n[亮点] 摊薄ROE>20% 且 扣非净利正增长 且 负债<60% 且 有股息: {len(bright)} 只")
print(bright.sort_values('摊薄ROE(%)', ascending=False).head(25).to_string(index=False))
