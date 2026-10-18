import h5py
import numpy as np


class ChannelDataSource:
    def __init__(
        self,
        hfname="../../../../../../Data/NREL/3D_CHANNEL/channel_5200_data_0_10000.h5"
    ):
        self.hfname = hfname
        with h5py.File(self.hfname, "r") as f:
            dset = f["data"]
            assert isinstance(dset, h5py.Dataset)
            shape = dset.shape
            self.fields = f["fields"][()]
            self.x = f["x"][()]
            self.y = f["y"][()]
            self.z = f["z"][()]
        self.dim_order = ["times", "fields", "x", "y", "z"]
        self.n_snapshots = shape[0]

    def __len__(self):
        return self.n_snapshots

    def __getitem__(self, key):
        with h5py.File(self.hfname, "r") as f:
            dset = f["data"]
            assert isinstance(dset, h5py.Dataset)
            data = np.array(dset[key])
        return data