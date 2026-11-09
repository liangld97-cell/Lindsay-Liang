import rqdatac
import pandas as pd
rqdatac.init()
from rqdatac import get_factor_return

# 近一年的全市场风格因子收益率（v1 模型）
fr = get_factor_return('20250801', '20260807',
                       factors=None, universe='whole_market',
                       method='implicit', model='v1',
                       industry_mapping='sws_2021')

# 只保留 11 个风格因子，去掉行业因子
style = ['beta', 'book_to_price', 'comovement', 'earnings_yield', 'growth',
         'leverage', 'liquidity', 'momentum', 'non_linear_size',
         'residual_volatility', 'size']
fr_style = fr[style].copy()

# 保存日度因子收益率
fr_style.to_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/factor_returns_1y.csv", encoding='utf-8-sig')

# 构建累计因子指数（复利累乘，起始=1）
index_curve = (1 + fr_style).cumprod()
index_curve.to_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/factor_index_1y.csv", encoding='utf-8-sig')

# 汇总
total = index_curve.iloc[-1] - 1
summary = total.sort_values(ascending=False)
print(f"因子收益率序列: {len(fr_style)} 个交易日, {fr_style.index.min().date()} ~ {fr_style.index.max().date()}\n")
print("=" * 52)
print("近一年风格因子累计收益排名（因子指数 -1）：")
print("=" * 52)
for name, val in summary.items():
    bar = '█' * int(abs(val) * 100 / 3)
    print(f"{name:<20} {val*100:>7.2f}%  {bar}")

print("\n最近30个交易日的动量因子累计指数：")
print(index_curve['momentum'].tail(30).round(4).to_string())
