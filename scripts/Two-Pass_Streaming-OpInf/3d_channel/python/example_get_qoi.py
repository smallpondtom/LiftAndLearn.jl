import qoi
from channel_data_source import ChannelDataSource

tidx = 100
data_source = ChannelDataSource()
test_snapshot = data_source[tidx]
z = data_source.z

qois = qoi.get_qois(test_snapshot, z)
print(qois["zprof"])
print(qois["utau"])