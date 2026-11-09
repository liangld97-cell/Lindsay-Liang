import rqdatac
import pandas as pd
import re
rqdatac.init()
from rqdatac import get_factor_return

def is_chinese(s):
    return bool(re.search(r'[一-鿿]', s))

# ============ v2 模型 ============
fr2 = get_factor_return('20250801', '20260807',
                        factors=None, universe='whole_market',
                        method='implicit', model='v2',
                        industry_mapping='sws_2021')

# 风格因子 = 英文列；行业因子 = 中文列
style2 = [c for c in fr2.columns if not is_chinese(c)]
fr2_style = fr2[style2].copy()
print(f"v2 风格因子数: {len(style2)}")
print("v2 风格因子列表:", style2)

idx2 = (1 + fr2_style).cumprod()
idx2.to_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/factor_index_v2_1y.csv", encoding='utf-8-sig')
fr2_style.to_csv(r"C:/Users/Lenovo/Desktop/rqdata-python-skill/factor_returns_v2_1y.csv", encoding='utf-8-sig')

total2 = (idx2.iloc[-1] - 1).sort_values(ascending=False)
print("\n" + "=" * 58)
print("近一年 v2 风格因子累计收益排名：")
print("=" * 58)
for name, val in total2.items():
    bar = '█' * int(abs(val) * 100 / 3)
    print(f"{name:<24} {val*100:>7.2f}%  {bar}")
