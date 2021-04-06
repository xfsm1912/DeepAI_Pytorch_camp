# -*- coding:utf-8 -*-
"""
@file name  : test_tensorboard.py
@author     : Jianhua Ma
@date       : 20210331
@brief      : test tensorboard can normally run.
"""
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# create the API for event file
# comment: folder suffix
writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):
    # record scalar
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()

