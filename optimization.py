import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from datetime import datetime
from itertools import product
from multiprocessing import Pool


from vnpy.trader.constant import Interval, Exchange
from vnpy_ctastrategy.base import BacktestingMode
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.strategies.hx_taking_strategy import HXTakingStrategy
from vnpy_ctastrategy.strategies.hx_td_taking_strategy import HXTDTakingStrategy
from vnpy_ctastrategy.strategies.hx_taking_strategy_double_threshold import HXTakingDoubleThresholdStrategy


# params = {
#     'training': {
#         "start": datetime(2021, 8, 2),
#         "end": datetime(2021, 9, 16),
#         "path": r"F:\HX\HFT-ML\exp\playground\test_rb_lightgbm\y_hat_training.pkl"
#         # "path": r"F:\HX\HFT-ML\exp\playground\test_rb\y_hat_training.pkl"
#     },
#     'valid': {
#         "start": datetime(2021, 9, 17),
#         "end": datetime(2021, 9, 30),
#         "path": r"F:\HX\HFT-ML\exp\playground\test_rb_lightgbm\y_hat_valid.pkl"
#         # "path": r"F:\HX\HFT-ML\exp\playground\test_rb\y_hat_valid.pkl"
#     },
#     'test': {
#         "start": datetime(2021, 10, 8),
#         "end": datetime(2021, 10, 28),
#         "path": r"F:\HX\HFT-ML\exp\playground\test_rb_lightgbm\y_hat_test.pkl"
#         # "path": r"F:\HX\HFT-ML\exp\playground\test_rb\y_hat_test.pkl"
#     }
# }

params = {
    'training': {
        "start": datetime(2021, 1, 4),
        "end": datetime(2021, 8, 4),  # datetime(2021, 8, 4),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_catboost\y_hat_training.pkl"
    },
    'valid': {
        "start": datetime(2021, 8, 5),
        "end": datetime(2021, 9, 30),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_catboost\y_hat_valid.pkl"
    },
    'test': {
        "start": datetime(2021, 10, 8),
        "end": datetime(2022, 1, 27),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_catboost\y_hat_test.pkl"
    }
}


def generate_settings(p) -> List[dict]:
    """"""
    keys = p.keys()
    values = p.values()
    products = list(product(*values))

    settings = []
    for p in products:
        setting = dict(zip(keys, p))
        settings.append(setting)

    return settings


def run_single_bkt(p):
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="ic1.CFFEX",  # rb.SHFE / ic1.CFFEX
        interval=Interval.TICK,
        start=params['training']['start'],
        end=params['training']['end'],
        rate=0.23 / 10000,
        slippage=0.,
        size=200,  # 10 / 200  合约乘数
        pricetick=.2,  # 1. / 0.2  最小变动价位
        capital=1_000_000,
        mode=BacktestingMode.TICK
    )
    print(p)
    engine.add_strategy(HXTakingDoubleThresholdStrategy, p)
    engine.callback = engine.strategy.on_tick
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()
    stats.update(p)

    gc.collect()
    return stats


if __name__ == '__main__':
    strategy_fixed_params = {
        'signal_df_path': params['training']['path'],
        'exchange': Exchange.CFFEX,
        'fixed_size': 1,
        'close_time_list': [(11, 30), (15, 50)]
    }
    strategy_params_grid = {
        'threshold': np.arange(50, 90, 5),
        'threshold_close': [-1.8e-4, 1.7e-4, 0.1e-5],  #  np.arange(0.02, 0.2, 0.02),
    }

    n_trails = 1
    for v in strategy_params_grid.values():
        n_trails *= len(v)
    print(f'总共需要回测{n_trails}次')

    param_list = generate_settings(strategy_params_grid)
    for param in param_list:
        param.update(strategy_fixed_params)

    with Pool(4) as p:  # 如果爆内存就调小进程池数量
        res_l = list(tqdm(p.imap(run_single_bkt, param_list), total=len(param_list)))

    # res_l = [run_single_bkt(param,) for param in tqdm(param_list)]
    # res_l = Parallel(n_jobs=4)(delayed(run_single_bkt)(e, param) for param in tqdm(param_list))
    res_df = pd.DataFrame.from_records(res_l)
    res_df.to_pickle('./optimization_result.pkl')
    res_df.to_csv('./optimization_result.csv')
