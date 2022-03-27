import gc

from vnpy.trader.optimize import OptimizationSetting
from vnpy.trader.constant import Interval, Exchange
from vnpy_ctastrategy.base import BacktestingMode
from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy_ctastrategy.strategies.hx_taking_strategy import HXTakingStrategy
from datetime import datetime


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
        "end": datetime(2021, 8, 4),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_lightgbm\y_hat_training.pkl"
    },
    'valid': {
        "start": datetime(2021, 8, 5),
        "end": datetime(2021, 9, 30),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_lightgbm\y_hat_valid.pkl"
    },
    'test': {
        "start": datetime(2021, 10, 8),
        "end": datetime(2022, 1, 27),
        "path": r"F:\HX\HFT-ML\exp\playground\test_ic_lightgbm\y_hat_test.pkl"
    }
}

for fold, p in params.items():
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="ic1.CFFEX",   # rb.SHFE / ic1.CFFEX
        interval=Interval.TICK,
        start=p['start'],
        end=p['end'],
        rate=0.3/10000,
        slippage=0.,
        size=200,                # 10 / 200
        pricetick=.2,            # 1. / 0.2
        capital=1_000_000,
        mode=BacktestingMode.TICK
    )
    engine.add_strategy(HXTakingStrategy,
                        {
                            "signal_df_path": p['path'],
                            "exchange": Exchange.CFFEX,  # Exchange.SHFE / Exchange.CFFEX
                            "fixed_size": 1,       # 50 / 1
                            "threshold": 2.35e-4,
                            # "close_time_list": [(10, 15), (11, 30), (15, 00), (23, 00)]  # RB
                            "close_time_list": [(11, 30), (15, 00)]   # IC
                        }
                        )
    engine.load_data()
    engine.callback = engine.strategy.on_tick

    # if fold == "training":
    #     setting = OptimizationSetting()
    #     setting.set_target("sharpe_ratio")
    #     setting.add_parameter("threshold", 1.9e-4, 2.4e-4, 1e-5)
    #
    #     engine.run_bf_optimization(setting)

    engine.run_backtesting()
    df = engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()

    del engine
    gc.collect()
