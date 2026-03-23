""" this is a 3D image registration algorithm developed using SimpleITK functions, which are entirely open source."""

from __future__ import print_function
import SimpleITK as sitk
import pydicom
import cv2
import glob
import os
import time
import numpy as np
import time
import pandas as pd
import argparse
import re

OUTPUT_DIR = 'Output'
dim = 512

"""the simpleITK algorithm uses a resample strategy and requires an interpolator, here we use the Nearest Neighbor"""
def resample(image, transform):
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    #sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def save_data(list_name,OUT_STAK,res_path):
    for i in range(len(OUT_STAK)):
        
        b=list_name[i].split('/')
        c=b.pop()
        d=c.split('.')
        cv2.imwrite(res_path+'/'+d[0][0:2]+str(int(d[0][2:5])+1)+'.png', OUT_STAK[i])
        
def command_iteration(filter):
    print(f"{filter.GetOptimizerIteration():3} = {filter.GetMetricValue():10.5f}")


"""the register function prepares an initializer setting some simpleITK parameters. In particular, it must receive 
- the reference, here called fix
- the moving, here called mov
- the transform function, which is the Euler3DTransform of SimpleITK library
- the initializer filter, which has the same role of estimate_initial in our codes and so, to have a fair comparison, it must be "MOMENTS". 
there are different similarity metrics so, to have a fair comparison, it must use the mutual information.
Differently from our code, the optimizer is one of the args passed when the algorithm is executed"""
def register(filename,fix, mov, metric, optimizer,input_path,moving_image,res_path, name_list, iterations):
    start_single_sw = time.time()
    t=0
    OUT_STAK=[]
    #print("Computing initial transform")
    x = sitk.CenteredTransformInitializer(fix,
                                          mov,
                                          sitk.Euler3DTransform(), 
                                          sitk.CenteredTransformInitializerFilter.MOMENTS)                                           

    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    if metric=="mse":
        registration_method.SetMetricAsMeanSquares()
    elif metric=="prz":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    elif metric=="mi":
        registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=256,varianceForJointPDFSmoothing=0)
    elif metric=="cc":
        registration_method.SetMetricAsCorrelation()
    else:
        return -100
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Optimizer settings.
    if optimizer=="gd":
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iterations, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    elif optimizer=="plone":
        registration_method.SetOptimizerAsOnePlusOneEvolutionary(numberOfIterations=iterations, epsilon=1.5e-4, initialRadius=1.01, growthFactor=-1.05, shrinkFactor=-0.99,seed=12345)
    elif optimizer=="pow":
        registration_method.SetOptimizerAsPowell(numberOfIterations=iterations, maximumLineIterations=iterations, stepLength=1, stepTolerance=1e-06, valueTolerance=0.00005)
    else:
        return -100


    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(x, inPlace=False)
    start_time = time.time()
    final_transform = registration_method.Execute(sitk.Cast(fix, sitk.sitkFloat32), 
                                                   sitk.Cast(mov, sitk.sitkFloat32))

    end_time= time.time()
    t=t+(end_time - start_time)
    
    FINALE=resample(mov, final_transform)

    print(FINALE.GetSize())
    naaa=sitk.GetArrayFromImage(FINALE)
    OUT_STAK=cv2.normalize(naaa,None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    end_single_sw = time.time()
    print('Final time: ', end_single_sw - start_single_sw)
    with open(filename, 'a') as file2:
        file2.write("%s\n" % (end_single_sw - start_single_sw)) 
    save_data(name_list,OUT_STAK,res_path)
    return t






"""This compute_wrapper does the same of the one in our framework algorithms but uses all simpleITK functions to have
 correct types and variables. In particular, type required is sitk.sitkFloat32.
 However, this function also reads images and prepares volumes, passed to the function register"""

def compute_wrapper(args, num_threads=1):
    curr_prefix = args.prefix+str(0)#+str(k)
    curr_ct = os.path.join(curr_prefix,args.ct_path)
    curr_pet = os.path.join(curr_prefix,args.pet_path)
    curr_res = os.path.join(curr_prefix,args.res_path)
    os.makedirs(curr_res,exist_ok=True)
    CT=glob.glob(curr_ct+'/*dcm')
    print(curr_pet)
    PET=glob.glob(curr_pet+'/*png')
    PET.sort(key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    CT.sort(key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    print(len(CT))
    print(len(PET))
    assert len(CT) == len(PET)
    times=[]
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(args.ct_path)
    dicom_names = np.asarray(dicom_names)
    dicom_names = sorted(dicom_names, key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    
    reader.SetFileNames(dicom_names)
    reader.SetOutputPixelType(sitk.sitkFloat32)

    fixed = reader.Execute()
    fixed1 = sitk.GetArrayFromImage(fixed)
    reader = sitk.ImageSeriesReader()
    png_names = [os.path.join(args.pet_path, f) for f in os.listdir(args.pet_path) if f.endswith(".png")]

    # Sort PNG files by their filename numbers for sequential stacking
    png_names = sorted(png_names, key=lambda var: [int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+', var)])

    # Read each PNG image individually and stack them as a 3D image
    image_slices = [sitk.ReadImage(png_name, sitk.sitkFloat32) for png_name in png_names]
    moving = sitk.JoinSeries(image_slices)

    #reader = sitk.ImageSeriesReader()
    #png_names = reader.GetPNGFileNames(args.pet_path)
    #png_names = np.asarray(png_names)
    #png_names = sorted(png_names, key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    #dicom_names = reader.GetGDCMSeriesFileNames(args.pet_path)
    #dicom_names = np.asarray(dicom_names)
    #dicom_names = sorted(dicom_names, key = lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
    #reader.SetFileNames(dicom_names)
    #reader.SetOutputPixelType(sitk.sitkFloat32)

    #moving = reader.Execute()
    #moving1 = sitk.GetArrayFromImage(moving)
    f=sitk.GetArrayFromImage(fixed)
    m=sitk.GetArrayFromImage(moving)
    fix=sitk.GetImageFromArray(f)
    mov=sitk.GetImageFromArray(m)
    t=register(args.filename,fix, mov, args.metric, args.optimizer,curr_res,moving,args.res_path, CT, args.iterations)
    


"""
This is the main function in which all parameters are passed. The only important element in this code is the call to compute_wrapper passing all arguments.
The code has been actually tested with 1 thread, even if the infrastructure is thought to be developed also considering more threads. This will surely be 
a Future Work.
"""

def main():

    parser = argparse.ArgumentParser(description='Iron software for IR onto a python env')
    parser.add_argument("-t", "--thread_number", nargs='?', help='Number of // threads', default=1, type=int)
    parser.add_argument("-mtr", "--metric", nargs='?', help='Metric to be tested, available mse, cc, mi, prz', default='mi')
    parser.add_argument("-opt", "--optimizer", nargs='?', help='optimizer to be tested, available plone, gd, pow', default='plone')
    parser.add_argument("-cp", "--ct_path", nargs='?', help='Path of the CT Images', default='./')
    parser.add_argument("-pp", "--pet_path", nargs='?', help='Path of the PET Images', default='./')
    parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results, with the / at the end', default='./')
    parser.add_argument("-f", "--filename", nargs='?', help='Name of the file in which to write times', default='test.csv')
    parser.add_argument("-px", "--prefix", nargs='?', help='prefix Path of patients folder, e.g. ~/faber_at_dac_/Data/Test/St,\
     it will be added a 0 after the prefix', default='./')
    parser.add_argument("-it", "--iterations", nargs='?', help='number of iterations to perform', default=100, type=int)
    
    print("CT_PATH: ", parser.parse_args().ct_path)
    print("PET_PATH: ", parser.parse_args().pet_path)
    args = parser.parse_args()
    num_threads = args.thread_number
    compute_wrapper(args, num_threads)


if __name__== "__main__":
    main()
