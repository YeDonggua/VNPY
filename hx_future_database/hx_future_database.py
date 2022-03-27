from typing import Dict, List
from datetime import datetime

import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../vnpy')
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview
)


class HXFutureDatabase(BaseDatabase):
    """HX 期货数据库接口"""
    def __init__(self) -> None:
        """"""
        self.db_path: str = SETTINGS["database.path"]
        self.db_dict = dict()

    def load_db_to_ram(self, symbol: str, exchange: Exchange) -> None:
        if symbol in self.db_dict:
            return
        self.db_dict[symbol] = pd.read_pickle(
            os.path.join(self.db_path, exchange.value, symbol+'.pkl')
        )

    def save_bar_data(self, bars: List[BarData]) -> bool:
        """保存k线数据"""
        return True  # 暂不支持bar数据

    def save_tick_data(self, ticks: List[TickData]) -> bool:
        """保存TICK数据"""
        return True  # 暂不支持写入tick数据

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """读取K线数据"""
        return []  # 暂不支持bar数据

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """读取Tick数据"""
        self.load_db_to_ram(symbol=symbol, exchange=exchange)

        # 转换时间格式
        start = np.datetime64(start)
        end = np.datetime64(end)

        # 读取数据DataFrame
        df = self.db_dict[symbol]
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

        if df.empty:
            return []

        df.set_index("datetime", inplace=True)

        # 转换为TickData格式
        ticks: List[TickData] = []

        for tp in df.itertuples():
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=tp.Index,
                name=tp.name,
                volume=tp.volume,
                turnover=np.nan,
                open_interest=tp.open_interest,
                last_price=tp.last_price,
                last_volume=np.nan,
                limit_up=np.nan,
                limit_down=np.nan,
                open_price=np.nan,
                high_price=np.nan,
                low_price=np.nan,
                pre_close=np.nan,
                bid_price_1=tp.bid_price_1,
                bid_price_2=tp.bid_price_2,
                bid_price_3=tp.bid_price_3,
                bid_price_4=tp.bid_price_4,
                bid_price_5=tp.bid_price_5,
                ask_price_1=tp.ask_price_1,
                ask_price_2=tp.ask_price_2,
                ask_price_3=tp.ask_price_3,
                ask_price_4=tp.ask_price_4,
                ask_price_5=tp.ask_price_5,
                bid_volume_1=tp.bid_volume_1,
                bid_volume_2=tp.bid_volume_2,
                bid_volume_3=tp.bid_volume_3,
                bid_volume_4=tp.bid_volume_4,
                bid_volume_5=tp.bid_volume_5,
                ask_volume_1=tp.ask_volume_1,
                ask_volume_2=tp.ask_volume_2,
                ask_volume_3=tp.ask_volume_3,
                ask_volume_4=tp.ask_volume_4,
                ask_volume_5=tp.ask_volume_5,
                localtime=tp.Index,
                gateway_name="DB"
            )
            ticks.append(tick)

        return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        return 0

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除Tick数据"""
        return 0

    def get_bar_overview(self) -> List[BarOverview]:
        """"查询数据库中的K线汇总信息"""
        return []