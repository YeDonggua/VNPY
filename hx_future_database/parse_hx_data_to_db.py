import os
import sys
import argparse
import pandas as pd

from glob import glob
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime, timedelta

sys.path.append('../vnpy')
from vnpy.trader.setting import SETTINGS


def t_to_datetime(t, date_str):
    """处理中金所IC数据的时间戳为datetime格式"""
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


def time_milsec_to_datetime(row, date_str):
    """处理上期所品种数据的时间戳为datetime格式"""
    time, milsec = row['time'], row['milsec']
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


def process_single_pickle_file(file_path):
    # if your os is unix-based please change the '\\' to '/'
    instrument = file_path.split('\\')[-1].split('_')[0]
    date_str = file_path.split('\\')[-1].split('_')[1].split('.')[0]
    df = pd.read_pickle(file_path, compression='gzip')

    df['symbol'] = instrument
    df['name'] = instrument
    if 't' in df.columns:
        df['datetime'] = df['t'].apply(t_to_datetime, args=(date_str,))
        del df['t']
        del df['mp']
    elif 'time' in df.columns and 'milsec' in df.columns:
        df['datetime'] = df[['time', 'milsec']].apply(time_milsec_to_datetime, args=(date_str,), axis=1)
        del df['time'], df['milsec']
    else:
        raise ValueError('date format might be wrong')
    df['date'] = df['datetime']
    return df


def main(args):
    file_list = glob(args.data_file_dir + '/' + f'*.pkl')

    total_list = Parallel(n_jobs=8)(delayed(process_single_pickle_file)(file) for file in tqdm(file_list))
    # total_list = [process_single_pickle_file(file) for file in tqdm(file_list)]
    df = pd.concat(total_list, ignore_index=True)
    df['exchange'] = args.exchange
    df.rename(
        columns={
            'last_pcx': 'last_price',
            'vol': 'volume',
            'cash': 'turnover',
            'oi': 'open_interest',
            'bp1': 'bid_price_1',
            'bp2': 'bid_price_2',
            'bp3': 'bid_price_3',
            'bp4': 'bid_price_4',
            'bp5': 'bid_price_5',
            'ap1': 'ask_price_1',
            'ap2': 'ask_price_2',
            'ap3': 'ask_price_3',
            'ap4': 'ask_price_4',
            'ap5': 'ask_price_5',
            'bv1': 'bid_volume_1',
            'bv2': 'bid_volume_2',
            'bv3': 'bid_volume_3',
            'bv4': 'bid_volume_4',
            'bv5': 'bid_volume_5',
            'av1': 'ask_volume_1',
            'av2': 'ask_volume_2',
            'av3': 'ask_volume_3',
            'av4': 'ask_volume_4',
            'av5': 'ask_volume_5'
        },
        inplace=True
    )
    df.sort_values(['symbol', 'datetime'], inplace=True)
    db_path = SETTINGS["database.path"]
    gb = df.groupby('symbol')
    for symbol, tmp_df in gb:
        Path(os.path.join(db_path, args.exchange)).mkdir(parents=True, exist_ok=True)
        tmp_df.reset_index(drop=True).to_pickle(os.path.join(db_path, args.exchange, symbol+'.pkl'))


def parse_args():
    parser = argparse.ArgumentParser(description='Data parsing')

    parser.add_argument('--data_file_dir', type=str, required=True)
    parser.add_argument('--exchange', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
