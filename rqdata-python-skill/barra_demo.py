import rqdatac
rqdatac.init()
from rqdatac import get_factor_exposure, get_descriptor_exposure, get_factor_return

# ========== 1) 单只股票的 BARRA 风格因子暴露度 ==========
print("=" * 60)
print("1) get_factor_exposure  <- 个股风格因子暴露度 (v1 模型)")
print("=" * 60)
exp = get_factor_exposure('600519.XSHG', '20260803', '20260807',
                          factors=None, model='v1', industry_mapping='sws_2021')
# 只看风格因子列（去掉行业因子列）
style_cols = [c for c in exp.columns if c not in exp.columns[exp.columns.str.contains('')] or True]
print(exp.loc[:, exp.columns[:11]].to_string())

# ========== 2) 全市场风格因子收益率（风格因子"指数"） ==========
print()
print("=" * 60)
print("2) get_factor_return  <- 全市场风格因子收益率 (近10个交易日)")
print("=" * 60)
ret = get_factor_return('20260727', '20260807', factors=None,
                        universe='whole_market', method='implicit',
                        model='v1', industry_mapping='sws_2021')
print(ret.iloc[:, :11].to_string())

# ========== 3) 细分风格因子暴露度 ==========
print()
print("=" * 60)
print("3) get_descriptor_exposure  <- 细分风格因子暴露度")
print("=" * 60)
desc = get_descriptor_exposure('600519.XSHG', '20260803', '20260807',
                               descriptors=['earnings_growth', 'book_leverage', 'sales_growth'],
                               model='v1')
print(desc.to_string())
