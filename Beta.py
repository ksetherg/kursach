import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import pandas_market_calendars as mcal
import time
from datetime import date, datetime
from Sigma import Sigma


class Beta:
    def __init__(self, dir_path):
        self.dir_path = Path(dir_path)

        self.start_date = '1996-01-01'
        self.end_date = '2018-11-30'
        self.gvkey_indexer = 'gvkey'
        self.date = 'datadate'

    def _get_timeline_(self):
        # get_timeline('1996-01-01', '2018-11-01') 5751 elements
        nyse = mcal.get_calendar('NYSE')  # New York Stock Exchange
        timeline = nyse.schedule(start_date=self.start_date, end_date=self.end_date)['market_open'].dt.date
        return timeline.values

    def _parse_dir_(self):
        file_list = sorted(self.dir_path.glob('*.h5'))
        return file_list

    @staticmethod
    def _str_to_date_(date_str, separator='-'):
        return datetime.strptime(date_str, '%Y{}%m{}%d'.format(*((separator,) * 2))).date()

    def _get_h5_data_(self, file_path):
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][:, :]
            timeline = np.vectorize(np.bytes_.decode)(h5f[self.date][:])
            self.gvkeys = np.vectorize(np.bytes_.decode)(h5f[self.gvkey_indexer][:])
        timeline = np.vectorize(self._str_to_date_)(timeline)
        self.timeline = np.sort(timeline)
        return data

    def _correct_timeline_(self):
        self.daily_timeline = self._get_timeline_()
        for i, date in enumerate(self.timeline):
            if date not in self.daily_timeline:
                idx = (np.abs(self.daily_timeline - date)).argmin()
                self.timeline[i] = self.daily_timeline[idx]

    def _transform_to_df_(self, data):
        self._correct_timeline_()
        df = pd.DataFrame(data=data.T, index=self.timeline, columns=self.gvkeys)
        df.reset_index(inplace=True)
        df.rename({'index': 'datadate'}, axis='columns', inplace=True)
        return df

    def _column_to_date_(self, df):
        df[self.date] = pd.to_datetime(df[self.date], format='%Y-%m-%d')
        df[self.date] = df[self.date].apply(lambda x: x.date())
        return df

    def _stretch_df_(self, df):
        base_for_strech = pd.DataFrame(data=self.daily_timeline, index=None, columns=[self.date])
        df_stretched = pd.merge(base_for_strech, df, on=[self.date], how='outer')
        df_stretched = df_stretched.sort_values(by=[self.date]).reset_index(drop=True)
        df_stretched.fillna(method='ffill', inplace=True)
        df_stretched.set_index('datadate', inplace=True)
        return df_stretched

    def _save_to_h5_(self, path, df):
        saver = Sigma(path, df.values, self.daily_timeline, self.gvkeys)
        saver._save_to_h5_()

    def evaluate(self):
        file_list = self._parse_dir_()
        for path in file_list:
            data = self._get_h5_data_(path)
            df = self._transform_to_df_(data)
            df_streched = self._stretch_df_(df)
            save_path = path.parents[1] / 'streched_splitted_data' / path.stem
            self._save_to_h5_(save_path, df_streched)
