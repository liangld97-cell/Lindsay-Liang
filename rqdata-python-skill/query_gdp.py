import rqdatac
rqdatac.init()
from rqdatac import econ

res = econ.get_factors(
    factors='国内生产总值GDP_累计同比',
    start_date='20190101',
    end_date='20260809',
)
print(res.to_string())
