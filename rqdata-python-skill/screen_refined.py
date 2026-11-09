import rqdatac
import pandas as pd
rqdatac.init()
from rqdatac import all_instruments, get_factor, is_st_stock

inst = all_instruments(type='CS')
codes = inst['order_book_id'].tolist()

# 1) 全市场因子：TTM 和 LY 双口径，摊薄ROE
factors = ['pe_ratio_ttm', 'pe_ratio_lyr',
           'return_on_equity_diluted_ttm', 'return_on_equity_diluted_lyr',
           'market_cap_3']
all_df = pd.DataFrame()
BATCH = 1000
for i in range(0, len(codes), BATCH):
    batch = codes[i:i + BATCH]
    all_df = pd.concat([all_df, get_factor(batch, factors)])

df = all_df.reset_index()
latest = df['date'].max()
df = df[df['date'] == latest].copy()
print(f"最近交易日: {latest.date()}")

# 2) ST 检测（最近交易日是否 ST）
st = is_st_stock(codes, latest, latest)
is_st = st.iloc[0]  # 该日的 True/False Series（index 为 order_book_id）
df['is_st'] = df['order_book_id'].map(is_st)

# 3) 四项筛选
m1 = ~df['is_st'].fillna(False)                                  # 非ST
m2 = df['market_cap_3'] >= 100e8                                  # 市值>=100亿
m3 = (df['pe_ratio_ttm'] > 0) & (df['pe_ratio_ttm'] < 20) \
   & (df['pe_ratio_lyr'] > 0) & (df['pe_ratio_lyr'] < 20)        # PE双口径
m4 = (df['return_on_equity_diluted_ttm'] > 0.15) \
   & (df['return_on_equity_diluted_lyr'] > 0.15)                 # 摊薄ROE双口径>15%
result = df[m1 & m2 & m3 & m4].copy()
result = result.sort_values('pe_ratio_ttm', ascending=True)

# 4) 整理输出
name_map = dict(zip(inst['order_book_id'], inst['symbol']))
result['名称'] = result['order_book_id'].map(name_map)
result['PE_ttm'] = result['pe_ratio_ttm'].round(2)
result['PE_lyr'] = result['pe_ratio_lyr'].round(2)
result['摊薄ROE_ttm(%)'] = (result['return_on_equity_diluted_ttm'] * 100).round(2)
result['摊薄ROE_lyr(%)'] = (result['return_on_equity_diluted_lyr'] * 100).round(2)
result['总市值(亿)'] = (result['market_cap_3'] / 1e8).round(0)

out = result[['order_book_id', '名称', 'PE_ttm', 'PE_lyr',
              '摊薄ROE_ttm(%)', '摊薄ROE_lyr(%)', '总市值(亿)']].rename(columns={'order_book_id': '代码'})

csv_path = r"C:/Users/Lenovo/Desktop/rqdata-python-skill/value_stocks_refined.csv"
out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n优化后筛选结果（非ST、市值>=100亿、PE双口径0-20、摊薄ROE双口径>15%）: 共 {len(out)} 只")
print("=" * 100)
pd.set_option('display.width', 150)
print(out.to_string(index=False))
