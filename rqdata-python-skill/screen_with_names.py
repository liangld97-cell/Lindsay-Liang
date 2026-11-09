import rqdatac
import pandas as pd
rqdatac.init()
from rqdatac import all_instruments, get_factor

inst = all_instruments(type='CS')
codes = inst['order_book_id'].tolist()

factors = ['pe_ratio_ttm', 'return_on_equity_ttm', 'market_cap_3']
all_df = pd.DataFrame()
BATCH = 1000
for i in range(0, len(codes), BATCH):
    batch = codes[i:i + BATCH]
    all_df = pd.concat([all_df, get_factor(batch, factors)])

df = all_df.reset_index()
df = df[df['date'] == df['date'].max()].copy()

mask = (df['return_on_equity_ttm'] > 0.15) & (df['pe_ratio_ttm'] > 0) & (df['pe_ratio_ttm'] < 20)
result = df[mask].dropna(subset=['pe_ratio_ttm', 'return_on_equity_ttm']).copy()
result = result.sort_values('pe_ratio_ttm', ascending=True)

# 合并股票名称
name_map = dict(zip(inst['order_book_id'], inst['symbol']))
result['名称'] = result['order_book_id'].map(name_map)
result['ROE(%)'] = (result['return_on_equity_ttm'] * 100).round(2)
result['市盈率'] = result['pe_ratio_ttm'].round(2)
result['总市值(亿)'] = (result['market_cap_3'] / 1e8).round(1)

out = result[['order_book_id', '名称', '市盈率', 'ROE(%)', '总市值(亿)']].rename(columns={'order_book_id': '代码'})

# 保存全量CSV
csv_path = r"C:/Users/Lenovo/Desktop/rqdata-python-skill/value_stocks_result.csv"
out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"共 {len(out)} 只，已保存到 {csv_path}")
print("=" * 70)
print("市盈率最低的前40名:")
pd.set_option('display.width', 120)
print(out.head(40).to_string(index=False))
