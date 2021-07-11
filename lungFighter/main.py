import mdset
import glob
import os
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
import sys

# center_xyz=(-102.84092514, 57.8809154545, -124.815520253)
# 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260
# ct = dsets.Ct(series_uid) 
# print(ct.getCtRawCandidate(center_xyz, (20,20,20))[0].shape)
mhd_path = '../../data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
ct_mhd = sitk.ReadImage(mhd_path)
ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
print(ct_a.clip(-1000, 1000, ct_a))

