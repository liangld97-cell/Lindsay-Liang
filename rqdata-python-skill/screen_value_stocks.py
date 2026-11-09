import rqdatac
import pandas as pd
rqdatac.init()
from rqdatac import all_instruments, get_factor

# 1) 获取全部A股
inst = all_instruments(type='CS')
codes = inst['order_book_id'].tolist()
print(f"全市场A股: {len(codes)} 只")

# 2) 分批发获取因子，避免单次请求过大
factors = ['pe_ratio_ttm', 'return_on_equity_ttm', 'market_cap_3']
all_df = pd.DataFrame()
BATCH = 1000
for i in range(0, len(codes), BATCH):
    batch = codes[i:i + BATCH]
    df = get_factor(batch, factors)
    all_df = pd.concat([all_df, df])
    print(f"已处理 {min(i + BATCH, len(codes))}/{len(codes)}")

# 3) 重置索引，取最近交易日
df = all_df.reset_index()
print("最近交易日:", df['date'].max())
df = df[df['date'] == df['date'].max()].copy()

# 4) 筛选：ROE > 15%（0.15），0 < 市盈率 < 20
mask = (df['return_on_equity_ttm'] > 0.15) & (df['pe_ratio_ttm'] > 0) & (df['pe_ratio_ttm'] < 20)
result = df[mask].dropna(subset=['pe_ratio_ttm', 'return_on_equity_ttm']).copy()
result = result.sort_values('pe_ratio_ttm', ascending=True)

# 5) 整理输出
result['ROE(%)'] = (result['return_on_equity_ttm'] * 100).round(2)
result['市盈率'] = result['pe_ratio_ttm'].round(2)
result['总市值(亿)'] = (result['market_cap_3'] / 1e8).round(1)
out = result[['order_book_id', '市盈率', 'ROE(%)', '总市值(亿)']].rename(columns={'order_book_id': '代码'})

print(f"\n筛选结果: ROE>15% 且 0<PE<20 共 {len(out)} 只")
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)
print(out.reset_index(drop=True))
