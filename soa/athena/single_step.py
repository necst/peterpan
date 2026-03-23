# /******************************************
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

# *MIT License
# *
# *Copyright (c) [2021] [Eleonora D'Arnese, Emanuele Del Sozzo, Davide Conficconi,  Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/
import random
import os
import re
import pydicom
import cv2
import numpy as np
import math
import glob
import time
import pandas as pd
from torch.multiprocessing import Pool, Process, set_start_method
import struct
import statistics
import argparse
import kornia
import torch

compute_metric = None
precompute_metric = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ref_vals = None
move_data = None
hist_dim = 256
dim = 512

def no_transfer(input_data):
    return input_data

def to_cuda(input_data):
    return input_data.cuda(non_blocking=True)


def transform(image, par, volume,mode):
    tmp_img = image.reshape((1, 1, *image.shape)).float()
    t_par = torch.unsqueeze(par, dim=0)
    img_warped = kornia.geometry.warp_affine3d(tmp_img, t_par, dsize=(volume, tmp_img.shape[3], tmp_img.shape[4]), flags=mode,align_corners = False)
    return img_warped


def compute_moments(img):
    moments = torch.empty(6, device=device)
    l = torch.arange(img.shape[0], device=device)
    moments[0] = torch.sum(img) # m00
    moments[1] = torch.sum(img * l) # m10
    moments[2] = torch.sum(img * (l**2)) # m20
    moments[3] = torch.sum(img * l.reshape((img.shape[0], 1)) ) # m01
    moments[4] = torch.sum(img * (l.reshape((img.shape[0], 1)))**2 ) # m02
    moments[5] = torch.sum(img * l * l.reshape((img.shape[0], 1))) # m11
    return moments

def to_matrix_complete_AROUND_XY_CENTER(vector_params):
    """
        vector_params contains tx, ty, tz for translation on x, y and z axes respectively
        and cosine of phi, theta, psi for rotations around x, y, and z axes respectively.
    """
    tx = vector_params[0]
    ty = vector_params[1]
    tz = vector_params[2]
    cos_phi = vector_params[3]
    cos_theta = vector_params[4]
    cos_psi = vector_params[5]
    sin_phi = -np.sqrt(1-(cos_phi**2))
    sin_theta = -np.sqrt(1-(cos_theta**2))
    sin_psi = -np.sqrt(1-(cos_psi**2))

    mat_offset_After = torch.eye(4, 4)
    mat_offset_After[0][3] = dim/2
    mat_offset_After[1][3] = dim/2
    mat_offset_After[2][3] = 0

    mat_offset_Before = torch.eye(4, 4)
    mat_offset_Before[0][3] = -dim/2
    mat_offset_Before[1][3] = -dim/2
    mat_offset_Before[2][3] = 0

    mat_translate = torch.eye(4, 4)
    mat_translate[0][3] = tx
    mat_translate[1][3] = ty
    mat_translate[2][3] = tz

    mat_rotate_z = torch.eye(4, 4)
    mat_rotate_z[0][0] = cos_psi
    mat_rotate_z[0][1] = -sin_psi
    mat_rotate_z[1][0] = sin_psi
    mat_rotate_z[1][1] = cos_psi

    mat_rotate_y = torch.eye(4, 4)
    mat_rotate_y[0][0] = cos_theta
    mat_rotate_y[0][2] = sin_theta
    mat_rotate_y[2][0] = -sin_theta
    mat_rotate_y[2][2] = cos_theta

    mat_rotate_x = torch.eye(4, 4)
    mat_rotate_x[1][1] = cos_phi
    mat_rotate_x[1][2] = -sin_phi
    mat_rotate_x[2][1] = sin_phi
    mat_rotate_x[2][2] = cos_phi

    mat_params = mat_offset_After @ mat_rotate_z @ mat_offset_Before @ mat_rotate_y @ mat_rotate_x @ mat_translate
    mat_params = torch.cat((mat_params[:3],mat_params[4:]))
    
    return (mat_params)

def estimate_rho(Ref_uint8s,Flt_uint8s, params, volume):
    tot_flt_avg_10 = 0
    tot_flt_avg_01 = 0
    tot_flt_mu_20 = 0
    tot_flt_mu_02 = 0
    tot_flt_mu_11 = 0
    tot_ref_avg_10 = 0
    tot_ref_avg_01 = 0
    tot_ref_mu_20 = 0
    tot_ref_mu_02 = 0
    tot_ref_mu_11 = 0
    tot_params1 = 0
    tot_params2 = 0
    tot_roundness = 0
    for i in range(0, volume):
        Ref_uint8 = Ref_uint8s[i, :, :]
        Flt_uint8 = Flt_uint8s[i, :, :]
        
        try:
            ref_mom = compute_moments(Ref_uint8)
            flt_mom = compute_moments(Flt_uint8)

            flt_avg_10 = flt_mom[1]/flt_mom[0]
            flt_avg_01 = flt_mom[3]/flt_mom[0]
            flt_mu_20 = (flt_mom[2]/flt_mom[0]*1.0)-(flt_avg_10*flt_avg_10)
            flt_mu_02 = (flt_mom[4]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_01)
            flt_mu_11 = (flt_mom[5]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_10)
            ref_avg_10 = ref_mom[1]/ref_mom[0]
            ref_avg_01 = ref_mom[3]/ref_mom[0]
            ref_mu_20 = (ref_mom[2]/ref_mom[0]*1.0)-(ref_avg_10*ref_avg_10)
            ref_mu_02 = (ref_mom[4]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_01)
            ref_mu_11 = (ref_mom[5]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_10)
            
            params[0] = ref_mom[1]/ref_mom[0] - flt_mom[1]/flt_mom[0]
            params[1] = ref_mom[3]/ref_mom[0] - flt_mom[3]/flt_mom[0]

            roundness=(flt_mom[2]/flt_mom[0]) / (flt_mom[4]/flt_mom[0])
            tot_flt_avg_10 += flt_avg_10
            tot_flt_avg_01 += flt_avg_01
            tot_flt_mu_20 += flt_mu_20
            tot_flt_mu_02 += flt_mu_02
            tot_flt_mu_11 += flt_mu_11
            tot_ref_avg_10 += ref_avg_10
            tot_ref_avg_01 += ref_avg_01
            tot_ref_mu_20 += ref_mu_20
            tot_ref_mu_02 += ref_mu_02
            tot_ref_mu_11 += ref_mu_11
            tot_params1 += params[0]
            tot_params2 += params[1]
            tot_roundness += roundness
        except:
             continue
        
    tot_flt_avg_10 = tot_flt_avg_10/volume
    tot_flt_avg_01 = tot_flt_avg_01/volume
    tot_flt_mu_20 = tot_flt_mu_20/volume
    tot_flt_mu_02 = tot_flt_mu_02/volume
    tot_flt_mu_11 = tot_flt_mu_11/volume
    tot_ref_avg_10 = tot_ref_avg_10/volume
    tot_ref_avg_01 = tot_ref_avg_01/volume
    tot_ref_mu_20 = tot_ref_mu_20/volume
    tot_ref_mu_02 = tot_ref_mu_02/volume
    tot_ref_mu_11 = tot_ref_mu_11/volume
    tot_params1 = tot_params1/volume
    tot_params2 = tot_params2/volume
    tot_roundness = tot_roundness/volume
    try: 
        rho_flt=0.5*torch.atan((2.0*flt_mu_11)/(flt_mu_20-flt_mu_02))
        rho_ref=0.5*torch.atan((2.0*ref_mu_11)/(ref_mu_20-ref_mu_02))
        delta_rho=rho_ref-rho_flt
        if math.fabs(tot_roundness-1.0)<0.3:
            delta_rho = 0
    except Exception as e:
        delta_rho = 0
    
    return torch.Tensor([delta_rho]), torch.Tensor([tot_params1]), torch.Tensor([tot_params2])

def precompute_mutual_information(Ref_uint8_ravel):
    
    href = torch.histc(Ref_uint8_ravel, bins=256)
    href /= Ref_uint8_ravel.numel()
    href=href[href>0.000000000000001]
    eref=(torch.sum(href*(torch.log2(href))))*-1
    
    return eref

def mutual_information(Ref_uint8_ravel, Flt_uint8_ravel, eref):
    
    if(device == "cuda:0"):
        idx_joint = torch.stack((Ref_uint8_ravel, Flt_uint8_ravel)).long()
        j_h_init = torch.sparse.IntTensor(idx_joint, ref_vals, torch.Size([hist_dim, hist_dim])).to_dense()/Ref_uint8_ravel.numel()
    else:
        idx_joint = torch.stack((Ref_uint8_ravel, Flt_uint8_ravel))
        j_h_init = my_squared_hist2d_t(idx_joint, hist_dim, 0, 255)/Ref_uint8_ravel.numel()
    
    j_h = j_h_init[j_h_init>0.000000000000001]
    entropy=(torch.sum(j_h*(torch.log2(j_h))))*-1
    
    hflt=torch.sum(j_h_init,axis=0) 
    hflt=hflt[hflt>0.000000000000001]
    eflt=(torch.sum(hflt*(torch.log2(hflt))))*-1
    
    mutualinfo=eref+eflt-entropy
    
    return(mutualinfo)

def my_squared_hist2d_t(sample, bins, smin, smax):
    D, N = sample.shape
    edges = torch.linspace(smin, smax, bins + 1, device=device)
    nbin = edges.shape[0] + 1
    
    # Compute the bin number each sample falls into.
    Ncount = D*[None]
    for i in range(D):
        Ncount[i] = torch.searchsorted(edges, sample[i, :], right=True)
    
    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[i, :] == edges[-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1
    
    # Compute the sample indices in the flattened histogram matrix.
    xy = Ncount[0]*nbin+Ncount[1]
           

    # Compute the number of repetitions in xy and assign it to the
    hist = torch.bincount(xy, None, minlength=nbin*nbin)
    
    # Shape into a proper matrix
    hist = hist.reshape((nbin, nbin))

    hist = hist.float()
    
    # Remove outliers (indices 0 and -1 for each dimension).
    hist = hist[1:-1,1:-1]
    
    return hist


def register_images(filename,Ref_uint8, Flt_uint8, volume,mode):
    optimal_params = [18.54458648,-12.30391042,0,1,1,0.93969262078] # 20 degrees
    params_trans=to_matrix_complete_AROUND_XY_CENTER(optimal_params)
    param = move_data(params_trans)

    start_single_sw = time.time()
    flt_transform = transform(Flt_uint8, param, volume,mode)
    torch.cuda.synchronize()
    end_single_sw = time.time()
    
    start_tot_time_mi= time.time()
    Ref_uint8_ravel = Ref_uint8.ravel().double()
    eref = precompute_mutual_information(Ref_uint8_ravel)
    eref_moved = move_data(eref)
    _ = mutual_information(Ref_uint8_ravel, flt_transform.ravel(), eref_moved)
    torch.cuda.synchronize()
    end_tot_time_mi = time.time()

    with open(filename, 'a') as file:
        TX_duration = end_single_sw - start_single_sw
        MI_duration = end_tot_time_mi - start_tot_time_mi
        tot_duration = TX_duration + MI_duration

        file.write(str(tot_duration) + ',' + str(TX_duration) + ',' + str(MI_duration) + '\n')
    return (flt_transform)


def save_data(OUT_STAK, name, res_path, volume):
    print("Going to save images: " + res_path)
    OUT_STAK = torch.reshape(OUT_STAK,(volume,1,512, 512))
    for i in range(len(OUT_STAK)):
        b=name[i].split('/')
        c=b.pop()
        d=c.split('.')
        # it requires the folder to be already created
        cv2.imwrite(os.path.join(res_path, d[0][0:2]+str(int(d[0][2:5]))+'.png'), kornia.tensor_to_image(OUT_STAK[i].cpu().byte())) 

def compute(CT, PET, name, curr_res, t_id, patient_id, filename, volume, mode, image_dimension):
    for _ in range(1):
        dim = image_dimension
        global ref_vals
        ref_vals = torch.ones(dim*dim*volume, dtype=torch.int, device=device)
        global move_data
        move_data = no_transfer if device=='cpu' else to_cuda
        refs = []
        flts = []
        couples = 0
        for c,ij in enumerate(zip(CT, PET)):
            i = ij[0]
            j = ij[1]  

            ref = cv2.imread(i)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            Ref_img = torch.tensor(ref.astype(np.int16), dtype=torch.int16, device=device)
            Ref_img[Ref_img==-2000]=1
            
            flt = cv2.imread(j)
            flt = cv2.cvtColor(flt, cv2.COLOR_BGR2GRAY)
            Flt_img = torch.tensor(flt.astype(np.int16), dtype=torch.int16, device=device)
            
            Ref_img = (Ref_img - Ref_img.min())/(Ref_img.max() - Ref_img.min())*255
            Ref_uint8 = Ref_img.round().type(torch.uint8)
                    
            Flt_img = (Flt_img - Flt_img.min())/(Flt_img.max() - Flt_img.min())*255
            Flt_uint8 = Flt_img.round().type(torch.uint8)
        
            refs.append(Ref_uint8)
            flts.append(Flt_uint8)
            couples = couples + 1
            if couples >= volume:
                break
            
        refs3D = torch.cat(refs)
        flts3D = torch.cat(flts)
        refs3D = torch.reshape(refs3D,(volume,dim,dim))
        flts3D = torch.reshape(flts3D,(volume,dim,dim))
        f_img=(register_images(filename, refs3D, flts3D, volume, mode))
        f_img_cpu = f_img.cpu()
        save_data(f_img_cpu,PET,curr_res,volume)


def compute_wrapper(args, num_threads=1):
    config=args.config
    
    for k in range(args.offset, args.patient):
        pool = []
        curr_prefix = args.prefix
        curr_ct = os.path.join(curr_prefix,args.ct_path)
        curr_pet = os.path.join(curr_prefix,args.pet_path)
        curr_res = os.path.join("",args.res_path)
        os.makedirs(curr_res,exist_ok=True)
        CT=glob.glob(curr_ct+'/*png')
        PET=glob.glob(curr_pet+'/*png')
        PET.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        CT.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        assert len(CT) == len(PET)
        images_per_thread = len(CT) // num_threads
        for i in range(num_threads):
            start = images_per_thread * i
            end = images_per_thread * (i + 1) if i < num_threads - 1 else len(CT)
            name = "t%02d" % (i)
            set_start_method('spawn')
            pool.append(Process(target=compute, args=(CT[start:end], PET[start:end], name, curr_res, i, k, args.filename,args.volume,args.mode,args.image_dimension)))
        for t in pool:
            t.start()
        for t in pool:
            t.join()

hist_dim = 256

def main():

    parser = argparse.ArgumentParser(description='Iron software for IR onto a python env')
    parser.add_argument("-pt", "--patient", nargs='?', help='Number of the patient to analyze', default=1, type=int)
    parser.add_argument("-o", "--offset", nargs='?', help='Starting patient to analyze', default=0, type=int)
    parser.add_argument("-cp", "--ct_path", nargs='?', help='Path of the CT Images', default='./')
    parser.add_argument("-pp", "--pet_path", nargs='?', help='Path of the PET Images', default='./')
    parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results', default='./output/')
    parser.add_argument("-t", "--thread_number", nargs='?', help='Number of // threads', default=1, type=int)
    parser.add_argument("-px", "--prefix", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-im", "--image_dimension", nargs='?', help='Target images dimensions', default=512, type=int)
    parser.add_argument("-c", "--config", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-dvc", "--device", nargs='?', help='Target device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument("-vol", "--volume", nargs='?', help='Volume',type = int, default=512)
    parser.add_argument("-f", "--filename", nargs='?', help='Filename', default="test.csv")
    parser.add_argument("-m", "--mode", nargs='?', help='interpolation mode', default="bilinear")


    args = parser.parse_args()
    num_threads=args.thread_number

    patient_number=args.patient
   

    global device
    device = args.device

    compute_wrapper(args, num_threads)


if __name__== "__main__":
    main()

















