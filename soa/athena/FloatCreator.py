# MIT License

# Copyright (c) 2025 Giuseppe Sorrentino, Paolo Salvatore Galfano, Davide Conficconi, Eleonora D'Arnese

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from skimage import exposure
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt
import pydicom as dicom
import glob
import pandas as pd
import numpy as np
import torch
import re
import pydicom
import faberImageRegistration as fir

def save_data(final_img):
    for i in range(len(final_img)):
        cv2.imwrite("Dataset/GeneratedFloat/IM" + str(i)+".png",final_img[i])




parser = argparse.ArgumentParser(description='Iron software for IR onto a python env')
parser.add_argument("-tx", "--tx", nargs='?', help='Translation over X axis', default=0,type=float)
parser.add_argument("-ty", "--ty", nargs='?', help='Translation over Y axis', default=0,type=float)
parser.add_argument("-tz", "--tz",  nargs = '?', help='Translation over Z axis',default=0,type=float)
parser.add_argument("-cosX", "--cosX", nargs='?', help="Rotation around X axis", default=100,type=float)
parser.add_argument("-cosY", "--cosY", nargs='?', help="Rotation around Y axis", default=100,type=float)
parser.add_argument("-cosZ", "--cosZ", nargs='?', help="Rotation around Z axis", default=100,type=float)

args = parser.parse_args()

couples = 0
refs=[]
volume = 246
curr_ct = 'Dataset/ST0/SE4'
CT=glob.glob(curr_ct + '/*dcm')
CT.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
for c,i in enumerate(CT):
    ref = pydicom.dcmread(i)
    Ref_img = torch.tensor(ref.pixel_array.astype(np.int16), dtype=torch.int16)
    Ref_img[Ref_img==-2000]=1
    Ref_img = (Ref_img - Ref_img.min())/(Ref_img.max() - Ref_img.min())*255
    Ref_uint8 = Ref_img.round().type(torch.uint8)
    refs.append(Ref_uint8)
    couples = couples +1
    if couples >= volume:
        break
refs3D = torch.cat(refs)
refs3D = torch.reshape(refs3D,(volume,512,512))
params = fir.to_matrix_complete([args.tx,args.ty,args.tz,args.cosX / 100,args.cosY / 100,args.cosZ / 100])
flt = fir.transform(refs3D,params,volume)
save_data(flt)