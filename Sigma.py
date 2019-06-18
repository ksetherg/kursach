import numpy as np
import h5py
import os

class Sigma:
    def __init__(self, path, data, timeline, gvkeys):
        assert isinstance(data, np.ndarray)
        assert isinstance(timeline, np.ndarray)
        assert isinstance(gvkeys, np.ndarray)

        self.save_path = path
        self.data = data
        self.timeline = self._array_into_bstr_(timeline)
        self.gvkeys = self._array_into_bstr_(gvkeys)

        self.shape_0, self.shape_1 = data.shape
        assert self.shape_0 == gvkeys.shape[0]
        assert self.shape_1 == timeline.shape[0]

        self.date = 'datadate'
        self.gvkey_indexer = 'gvkey'


    def _array_into_bstr_(self, array):
        str_arr = np.vectorize(str)(array)
        bstr_arr = np.vectorize(str.encode)(str_arr)
        return bstr_arr

    def _h5_file_create_(self):
        '''create h5 file with fixed stucture'''
        with h5py.File(self.save_path, 'w') as h5f:
            '''timeline по умолчанию заполняем nan'''
            h5f.create_dataset(self.date,
                               self.timeline.shape,
                               maxshape=self.timeline.shape,
                               dtype='<S10',
                               compression="gzip",
                               compression_opts=9)
            '''gvkey по умолчанию заполняем nan'''
            h5f.create_dataset(self.gvkey_indexer,
                               self.gvkeys.shape,
                               maxshape=self.gvkeys.shape,
                               dtype='<S10',
                               compression="gzip",
                               compression_opts=9)
            '''data по умолчанию заполняем nan'''
            h5f.create_dataset('data',
                               self.data.shape,
                               maxshape=self.data.shape,
                               dtype='<f4',
                               fillvalue=np.nan,
                               compression="gzip",
                               compression_opts=9)

    def _protect_h5_creater_(self):
        '''clever saver with deleting previous version of file'''
        if not os.path.exists(self.save_path):
            self._h5_file_create_()
        else:
            os.remove(self.save_path)
            print('Deleted: ', self.save_path)
            self._h5_file_create_()

    def _save_to_h5_(self):
        '''create and save data to h5 file'''
        self._protect_h5_creater_()
        with h5py.File(self.save_path, 'a') as h5f:
            h5f['data'][:, :] = self.data
            h5f[self.date][:] = self.timeline
            h5f[self.gvkey_indexer][:] = self.gvkeys