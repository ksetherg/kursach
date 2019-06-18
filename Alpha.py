"""Parse huge csv files and save fields binary"""
import pandas as pd
import numpy as np
from pathlib import Path
from Sigma import Sigma


class Alpha:
    def __init__(self, path, keys):
        self.gvkeys = None
        self.timeline = None
        self.process_fields = keys
        self.path_to_csv_file = path

        self.gvkey_indexer = 'gvkey'
        self.date = 'datadate'
        self.full_keys = [self.gvkey_indexer, self.date] + keys
        self.chunk_size = 100000

    def _load_dataframe_(self):
        """Chunk Iterator"""
        df_iter = pd.read_csv(self.path_to_csv_file,
                         index_col=None,
                         low_memory=False,
                         iterator=True,
                         chunksize=self.chunk_size,
                         usecols=self.full_keys,
                         engine='c')
        return df_iter

    def _str_to_date_(self, df):
        df[self.date] = pd.to_datetime(df[self.date], format='%Y%m%d')
        df[self.date] = df[self.date].apply(lambda x: x.date())
        return df

    def _prepare_data_(self):
        df_iter = self._load_dataframe_()
        print('Starting concatenating df...')
        df = pd.concat(df_iter, ignore_index=True)
        df = self._str_to_date_(df)
        print('Starting dropping duplicates')
        df = df.groupby(self.gvkey_indexer).apply(lambda chunk: chunk.drop_duplicates(subset=[self.date])).reset_index(drop=True)
        return df

    def _stretch_df_(self):
        df = self._prepare_data_()
        '''if gvkey str, need to be transformed'''
        self.unique_gvkey = np.sort(df[self.gvkey_indexer].unique())
        self.timeline = np.sort(df[self.date].unique())

        index = pd.MultiIndex.from_product([self.unique_gvkey, self.timeline], names=[self.gvkey_indexer , self.date])
        base_for_strech = pd.DataFrame(index=index).reset_index()
        base_for_strech[self.date] = base_for_strech[self.date].apply(lambda x: x.date())
        print('Starting streching df...')
        df_streched = pd.merge(base_for_strech, df, on=[self.date, self.gvkey_indexer ], how='outer')
        df_streched = df_streched.sort_values(by=[self.gvkey_indexer , self.date]).reset_index(drop=True)
        return df_streched

    def _to_numpy_arr(self, np_arr_base, chunk_name, chunk):
        indexer = np.where(self.unique_gvkey == chunk_name)[0][0]
        np_arr_base[indexer] = chunk.to_numpy(dtype=np.float32)
        return np_arr_base

    def _process_df_(self):
        df_streched = self._stretch_df_()
        df_streched = df_streched.fillna(method='ffill')
        np_arr_base = np.empty((self.unique_gvkey.size, self.timeline.size, len(self.process_fields)))
        print('Starting transforming into numpy arr...')
        df_streched.groupby(self.gvkey_indexer).apply(lambda chunk: self._to_numpy_arr(np_arr_base, chunk.name, chunk[self.process_fields]))
        return np_arr_base

    def _evaluate_saving_(self, all_data_arr):
        for i, key in enumerate(self.process_fields):
            print('Saving %s field' % (key,))
            file_path = Path(self.path_to_csv_file).parent / 'raw_splitted_data' / (key + '.h5')
            saver = Sigma(file_path, all_data_arr[:, :, i], self.timeline, self.gvkeys)
            saver._save_to_h5_()

    def parse_csv(self):
        all_data_arr = self._process_df_()
        self._evaluate_saving_(all_data_arr)
        print('Done.')