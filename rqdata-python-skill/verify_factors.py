import rqdatac
rqdatac.init()
from rqdatac import all_instruments, get_factor

# 1) 获取全部A股
inst = all_instruments(type='CS')
print("股票总数:", len(inst))
print("代码格式示例:", inst['order_book_id'].head(5).tolist())

# 2) 用几只已知股票验证因子名和单位
codes = ['600519.XSHG', '600036.XSHG']  # 茅台、招行
df = get_factor(codes, ['pe_ratio_ttm', 'return_on_equity_ttm'])
print("\n--- 验证结果 ---")
print(df)
