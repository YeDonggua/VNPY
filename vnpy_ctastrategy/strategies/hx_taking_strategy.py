import pandas as pd
import pickle5 as pickle
from datetime import datetime, timedelta
from vnpy.trader.constant import Exchange, Status, Direction, Offset
from vnpy_ctastrategy import (
    TargetPosTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)


def t_to_datetime(row):
    """处理中金所IC数据的时间戳为datetime格式"""
    t, date_str = row['t'], row['date']
    time_str = str(int(t)).zfill(9)
    return datetime(
        year=int(date_str[:4]),
        month=int(date_str[4:6]),
        day=int(date_str[6:]),
        hour=int(time_str[:2]),
        minute=int(time_str[2:4]),
        second=int(time_str[4:6]),
        microsecond=int(time_str[6:])*1000
    )


def time_milsec_to_datetime(row):
    """处理上期所品种数据的时间戳为datetime格式"""
    time, milsec, date_str = row['time'], row['milsec'], row['date']
    time_str = str(time).zfill(6)
    dt = datetime(
            year=int(date_str[:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:]),
            hour=int(time_str[:2]),
            minute=int(time_str[2:4]),
            second=int(time_str[4:6]),
            microsecond=milsec*1000
    )
    if dt.hour < 8:
        # 部分品种夜盘结束时间是次日凌晨，日期需+1
        dt = dt + timedelta(days=1)
    return dt


class HXTakingStrategy(TargetPosTemplate):
    """"""

    author = "Ye Wenxuan"

    threshold = 1e-4
    fixed_size = 1

    target_pos = 0
    entry_tick = None

    parameters = [
        "threshold",
        "fixed_size"
    ]
    variables = [
        "target_pos",
        "entry_tick"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.signal_dict = None
        self.setting = setting

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        try:
            signal_df = pd.read_pickle(self.setting['signal_df_path'])
        except ValueError:
            with open(self.setting['signal_df_path'], 'rb') as f:
                signal_df = pickle.load(f)
        # if self.setting['exchange'] == Exchange.SHFE:
        signal_df['datetime'] = signal_df[['time', 'date', 'milsec']].apply(time_milsec_to_datetime, axis=1)
        # elif self.setting['exchange'] == Exchange.CFFEX:
        #     signal_df['datetime'] = signal_df[['t', 'date']].apply(t_to_datetime, axis=1)
        signal_df.set_index('datetime', inplace=True)
        self.signal_dict = signal_df[['pred']].to_dict('index')

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def calculate_target_pos(self, tick):
        # 判断是否快收盘了
        for h, m in self.setting['close_time_list']:
            close_datetime = datetime(year=tick.datetime.year,
                                      month=tick.datetime.month,
                                      day=tick.datetime.day,
                                      hour=h,
                                      minute=m,
                                      second=0)
            if abs((close_datetime - tick.datetime).seconds) <= 60:
                self.set_target_pos(0)
                return

        # 10秒即平
        if self.target_pos != 0 and (tick.datetime - self.entry_tick.datetime).seconds >= 10:
            self.set_target_pos(0)
            return

        # 读取事先算好的信号
        pred = self.signal_dict.get(tick.datetime, None)
        if not pred:
            return
        signal = pred['pred']

        if signal > self.threshold:
            self.set_target_pos(self.fixed_size)
            self.entry_tick = tick
        elif signal < -self.threshold:
            self.set_target_pos(-self.fixed_size)
            self.entry_tick = tick

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        super(HXTakingStrategy, self).on_tick(tick)
        self.calculate_target_pos(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        super(HXTakingStrategy, self).on_order(order)

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
