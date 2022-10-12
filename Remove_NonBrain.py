# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

import os
from os.path import join
import shutil
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy,color_fa



bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                   'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                   'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
            "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
            "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
            "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
            "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
            "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
            "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
            "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
            "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
            "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
            "620434", "613538"]##100
#
# subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
#         "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
#         "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
#         "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
#         "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#         "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
#         "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
#         "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
#         "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
#         "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
#         "620434", "613538", "601127", "599671", "599469"]  #HCP103


# subjects = ["992774", "991267", "987983", "984472", "983773",  "978578", "965771", "965367", "959574",
#         "958976", "957974", "932554", "930449", "922854", "917255", "912447", "910241",
#         "904044", "901442", "901139",  "899885", "898176", "896879", "896778", "894673", "889579",
#         "877269", "877168", "872158", "871964", "871762", "865363", "861456", "859671",
#         "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#         "816653", "802844", "792564", "789373", "786569", "784565", "782561", "779370",
#         "771354", "770352", "765056", "761957", "759869", "756055", "751348", "748662",
#         "748258", "742549", "734045", "732243", "729557", "729254", "715647", "709551", "705341",
#         "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
#         "673455", "672756", "665254", "654754", "645551", "638049", "627549", "623844", "622236",
#         "613538", "599671", "599469"]  #HCP91----gradient288




bundles_tt = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
bundles_index = [14, 15, 18, 19, 29, 30, 31, 32, 43, 44]
subjects_tt = ['A00','A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']


##---copy the bundle_masks_72.nii.gz into the HCP_training_COPY document----
def copy_bundles_label():
    subjects = ['A00','A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']

    print(len(subjects))
    # ori_dir = '/data4/wanliu/HCP_Wasser_Clinical/HCP_for_training_COPY'
    # new_dir = '/data4/wanliu/HCP_Wasser_Clinical_MoreGrad/HCP_for_training_COPY'
    # ori_dir = '/data1/qilu/HCP_wassertha/HCP_for_training_COPY'
    # new_dir = '/data4/wanliu/HCP_Wasser_LessGrad_641k/HCP_for_training_COPY'

    ori_dir = '/data4/wanliu/HCP_1.25mm_270/HCP_AtlasFSL/40sub/BT_1.7_270'
    new_dir = '/data4/wanliu/TractSeg_Run/Compare_results/label_embed/tmi2021embed/results_bt_2ndNIMG/BT_1.7_270/atlas'

    for sub in subjects:
        if os.path.exists(join(new_dir, sub))==0:
            os.makedirs(join(new_dir, sub))
        # for target_file in ["bundle_masks_72.nii.gz"]:
        for target_file in ["Segment_40subject.nii.gz"]:

            ori_file = join(ori_dir, sub, target_file)
            new_file = join(new_dir, sub, target_file)
            shutil.copyfile(ori_file, new_file)

##---copy the generated peaks into the HCP_training_COPY document----
def copy_generated_peaks():
    # ori_dir = '/data4/wanliu/HCP105_initdata/HCP_Low_MoreGrad'
    # new_dir = '/data4/wanliu/HCP_Wasser_Clinical_MoreGrad/HCP_for_training_COPY'
    # ori_dir = '/data4/wanliu/HCP105_initdata/HCP_High_LessGrad'
    # new_dir = '/data4/wanliu/HCP_Wasser_LessGrad/HCP_for_training_COPY'
    # ori_dir = '/data4/wanliu/HCP105_initdata/HCP_High_LessGrad_361k2k_b01'
    # new_dir = '/data4/wanliu/HCP_Wasser_LessGrad_361k2k_b01/HCP_for_training_COPY'
    # ori_dir = '/data4/wanliu/TT_initdata/TT_init_LessGrad'
    # new_dir = '/data4/wanliu/TT_LessGrad/HCP_for_training_COPY'
    ori_dir = '/data4/wanliu/HCP105_initdata/HCP_High_LessGrad_64'
    new_dir = '/data4/wanliu/HCP_Wasser_LessGrad_641k/HCP_for_training_COPY'

    for sub in subjects:
        if os.path.exists(join(new_dir, sub)) == 0:
            os.makedirs(join(new_dir, sub))
        for target_file in ["peaks.nii.gz"]:
            ori_file = join(ori_dir, sub,'tractseg_output',target_file)
            new_file = join(new_dir, sub, "mrtrix_peaks.nii.gz")
            shutil.copyfile(ori_file, new_file)
            print('done1')

##----------- remove the non-brain region of peaks and labels in the HCP_training_COPY file to HCP_preproc----------
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def remove_nonbrain_area_HCP():
    ori_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_786569_4/MtractoutYChange/HCP_for_training_COPY'
    new_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_daug/Oneshot_786569_4/MtractoutYChange/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    # target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_72.nii.gz']
    target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_4.nii.gz']
    subjects=os.listdir(ori_dir)
    for sub in subjects:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
            if os.path.exists(join(new_dir, sub))==0:
                os.makedirs(join(new_dir, sub))
            nib.save(data_nii, new_path)


def remove_nonbrain_area_HCP_update():
    # sub_list=["792766",'623844','690152','620434']
    sub_list=['623844','620434']

    tract_num=4
    # daug_list = ['CutoutYChange','CutoutYKeep','MtractoutYChange','MtractoutYKeep']
    daug_list = ['CutoutYChange']

    for subj in sub_list:
        for daug in daug_list:
            root_dir='/data4/wanliu/HCP_2.5mm_341k/HCP_daug'
            ori_dir = join(root_dir,'Oneshot_'+subj+'_'+str(tract_num),daug,'HCP_for_training_COPY')
            new_dir = join(root_dir,'Oneshot_'+subj+'_'+str(tract_num),daug,'HCP_preproc')
            print(ori_dir, new_dir)
            help_file = 'mrtrix_peaks.nii.gz'
            target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_'+str(tract_num)+'.nii.gz']
            subjects=os.listdir(ori_dir)
            for sub in subjects:
                print(sub)
                for target_file in target_file_list:
                    ## extract crop area
                    help_path = join(ori_dir, sub, help_file)
                    help_data = np.nan_to_num(nib.load(help_path).get_fdata())
                    bbox = get_bbox_from_mask(help_data)

                    ## load the data
                    ori_path = join(ori_dir, sub, target_file)
                    new_path = join(new_dir, sub, target_file)
                    large_nii = nib.load(ori_path)
                    large_data = np.nan_to_num(large_nii.get_fdata())
                    affine = large_nii.affine

                    ## crop non-brain area
                    data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
                    data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
                    if os.path.exists(join(new_dir, sub))==0:
                        os.makedirs(join(new_dir, sub))
                    nib.save(data_nii, new_path) 



def remove_nonbrain_area_TT():
    ori_dir = '/data4/wanliu/BT_1.7mm_270/TT_daug/Oneshot_C04_4/MtractoutYChange/HCP_for_training_COPY'
    new_dir = '/data4/wanliu/BT_1.7mm_270/TT_daug/Oneshot_C04_4/MtractoutYChange/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_4.nii.gz']

    sub_list=os.listdir(ori_dir)
    for sub in sub_list:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)

            if os.path.exists(join(new_dir, sub))==0:
                os.makedirs(join(new_dir, sub))

            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            if target_file=='mrtrix_peaks.nii.gz':
                data = data.astype(np.float32)
            else:
                data = data.astype(np.uint8)

            data_nii = nib.Nifti1Image(data, affine = affine)
            nib.save(data_nii, new_path)
            print(new_path)



## 40 AD subjects
BT_AD_SUBJECTS = ["AD0023", "AD0024","AD0025", "AD0026", "AD0027", "AD0028","AD0029","AD0030","AD0031", "AD0032",
                  "AD0033", "AD0034","AD0035", "AD0036", "AD0037", "AD0038","AD0039","AD0040","AD0041", "AD0042",
                  "AD0043", "AD0044","AD0045", "AD0046", "AD0047", "AD0048","AD0049","AD0050","AD0051", "AD0052",
                  "AD0053", "AD0054","AD0055", "AD0056", "AD0057", "AD0058","AD0059","AD0060","AD0061", "AD0062"]
## 40 HC subjects
BT_HC_SUBJECTS = ["HC0158", "HC0159", "HC0160", "HC0161", "HC0162", "HC0163", "HC0164", "HC0165", "HC0166", "HC0167",
                  "HC0168", "HC0169", "HC0170", "HC0171", "HC0172", "HC0173", "HC0174", "HC0175", "HC0176", "HC0177",
                  "HC0178", "HC0179", "HC0180", "HC0181", "HC0182", "HC0183", "HC0184", "HC0185", "HC0186", "HC0187",
                  "HC0188", "HC0189", "HC0190", "HC0191", "HC0192", "HC0193", "HC0194", "HC0195", "HC0196", "HC0197"]

def remove_nonbrain_area_atlasfsl():
    ori_dir1 = '/data4/wanliu/HCP_1.25mm_901k_18b0/HCP_for_training_COPY'
    ori_dir2 = '/data4/wanliu/HCP_1.25mm_270/HCP_AtlasFSL/train_output_file'
    type = ''
    new_dir = '/data4/wanliu/HCP_1.25mm_901k_18b0/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    target_file_list = ['bundle_masks_new10_atlasfsl.nii.gz']
    subjects=os.listdir(ori_dir1)
    # subjects=BT_HC_SUBJECTS

    for sub in subjects:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir1, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir2, sub, target_file)

            if target_file == 'peaks.nii.gz':
                new_path = join(new_dir, type+sub, 'mrtrix_peaks.nii.gz')
            else:
                new_path = join(new_dir, type+sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
            if os.path.exists(join(new_dir, type+sub))==0:
                os.makedirs(join(new_dir, type+sub))
            nib.save(data_nii, new_path)






def remove_nonbrain_area_HCP1():
    ori_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY_Spottune'
    new_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    # target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_72.nii.gz']
    target_file_list = ['bundle_masks_WarmWTYKeep_6.nii.gz', 'bundle_masks_pseudo_60.nii.gz']
    # subjects=os.listdir(ori_dir)
    for sub in ["845458"]:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
            if os.path.exists(join(new_dir, sub))==0:
                os.makedirs(join(new_dir, sub))
            nib.save(data_nii, new_path)




def remove_nonbrain_area_TT1():
    ori_dir = '/data4/wanliu/BT_1.7mm_270/HCP_for_training_COPY_Spottune'
    new_dir = '/data4/wanliu/BT_1.7mm_270/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    # target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_72.nii.gz']
    target_file_list = ['bundle_masks_WarmWTYKeep_6.nii.gz', 'bundle_masks_pseudo_60.nii.gz']
    # subjects=os.listdir(ori_dir)
    for sub in ["A00"]:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            if target_file=='mrtrix_peaks.nii.gz':
                data = data.astype(np.float32)
            else:
                data = data.astype(np.uint8)

            data_nii = nib.Nifti1Image(data, affine = affine)
            nib.save(data_nii, new_path)
            print(new_path)

if __name__ == "__main__":
    # copy_bundles_label()
    # copy_generated_peaks()
    # remove_nonbrain_area_HCP()
    # remove_nonbrain_area_HCP_update()

    # remove_nonbrain_area_TT()
    # remove_nonbrain_area_atlasfsl()

    remove_nonbrain_area_HCP1()
    # remove_nonbrain_area_TT1()

