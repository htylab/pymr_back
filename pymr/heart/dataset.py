import nibabel
import numpy as np
from ..heart import ahaseg
from .cine import get_frame
import matplotlib.pyplot as plt
from os.path import join, basename

def get_loc(num, got_apex):
    loc = dict()
    if got_apex> 0:
        #got_apex = True
        mid = num//3
        basal = mid
        apical = mid
        if num % 3 == 1:
            mid = mid + 1
        elif num % 3 == 2:
            basal += 1
            mid += 1
        
        loc = [1]*basal + [2]*mid + [3]*apical
        #loc[3] = [1]*1 + [2]*1 + [3]*1
        #loc[4] = [1]*1 + [2]*2 + [3]*1
        #loc[5] = [1]*2 + [2]*2 + [3]*1
        #loc[6] = [1]*2 + [2]*2 + [3]*2
        #loc[7] = [1]*2 + [2]*3 + [3]*2


    else:
        
        apex = num // 7

        num = num - apex

        mid = num//3
        basal = mid
        apical = mid
        if num % 3 == 1:
            mid = mid + 1
        elif num % 3 == 2:
            basal += 1
            mid += 1
            
        loc = [1]*basal + [2]*mid + [3]*apical + [4]*apex

    
    return loc

def get_slice_label(heart_xyzt):
    #basal: 1, 5
    #mid: 2
    #apical: 3
    #apex: 4
    # 4 and 5 are slices without RV mask
    heart = heart_xyzt.copy()
    curve = np.zeros((heart.shape[2],))
    sys_frame, dia_frame = get_frame(heart)
    for slicen in range(heart.shape[2]):    
        heart_mask_xyt = heart[:, :, slicen, :]
        if np.unique(heart_mask_xyt[..., sys_frame]).sum() + np.unique(heart_mask_xyt[..., dia_frame]).sum() == 12:
            curve[slicen] = 1

    curve_and = curve.astype(np.int)
    curve_or = (np.sum(heart, axis=(0, 1, 3)) > 0).astype(np.int)
    curve_diff = curve_or - curve_and
    diff = np.diff(np.sum(heart==1, axis=(0, 1, 3))) * curve_and[1:]
    diff = np.median(diff[curve_and[1:] > 0])

    slice_label = curve_and * 0
    mid_point = curve_and.size//2

    curve_apex = curve_diff.copy()
    curve_basal = curve_diff.copy()

    if diff < 0:
        basal_first = True

        curve_apex[:mid_point] = 0
        curve_basal[mid_point:] = 0
    else:
        basal_first = False
        curve_apex[mid_point:] = 0
        curve_basal[:mid_point] = 0

    got_apex = np.sum(curve_apex) > 0

    if basal_first:
        slice_label[curve_and > 0] = get_loc(np.sum(curve_and), got_apex)
        slice_label[curve_apex > 0] = 4
        slice_label[curve_basal > 0] = 11
    else:
        slice_label[curve_and > 0] = get_loc(np.sum(curve_and), got_apex)[::-1]
        slice_label[curve_apex > 0] = 4
        slice_label[curve_basal > 0] = 11
    #print(slice_label)
    return slice_label

def proc_basal_without_RV(mask1, mask5):
    if np.unique(mask5).sum() == 6:
        seg = ahaseg.get_seg(mask5, nseg=6)
        return seg
    
    # find nearest point with ahaseg
    xx5, yy5 = np.nonzero(mask5==2)
    xx1, yy1 = np.nonzero(mask1)
    seg5 = mask5 * 0
    for ii in range(xx5.size):
        index = np.sqrt((xx1 - xx5[ii])**2 + (yy1 - yy5[ii])**2).argmin()
        seg5[xx5[ii], yy5[ii]] = mask1[xx1[index], yy1[index]]
    return seg5


def convert(f, dataset='MMS', result_dir=None):
    if isinstance(f, str):
        temp = nibabel.load(f)
        heart = temp.get_fdata()
        affine = temp.affine
        file_input = True
    else:
        heart = f.copy()
        file_input = False
    
    new_f = basename(f)
    if dataset=='MMS':
        new_f = basename(f).replace('_gt', '_lab19')
        
    elif dataset=='ACDC':
        heart_temp = heart.copy()
        heart_temp[heart==1] = 3
        heart_temp[heart==3] = 1
        heart = heart_temp.copy()
        del heart_temp

        

    slice_label = get_slice_label(heart)
    #print(slice_label)
    offset = dict()
    offset[1] = 0
    offset[2] = 6
    offset[3] = 12

    count = -1
    heart_aha17_4d = heart  * 0
    for ii in range(slice_label.size):
        #print(ii)
        #print(slice_label[ii])
        count = count + 1
        if (slice_label[ii] == 0) or (slice_label[ii] == 11):
            continue

        if slice_label[ii] == 4:
            temp = heart[:, :, ii, :].copy()
            temp[temp==2] = 17
            temp[temp==1] = 0
            temp[temp==3] = 0
            heart_aha17_4d[:, :, ii, :] = temp
            continue


        heart_xyt = heart[:, :, ii, :].copy()

        sys_frame, dia_frame = get_frame(heart_xyt)

        #print(dia_frame, sys_frame)
        heart_xy_dia = heart_xyt[..., dia_frame]
        heart_xy_sys = heart_xyt[..., sys_frame]
        if slice_label[ii] == 3:
            nseg = 4
        else:
            nseg = 6

        dia_seg = ahaseg.get_seg(heart_xy_dia, nseg)
        sys_seg = ahaseg.get_seg(heart_xy_sys, nseg)
        dia_seg[dia_seg > 0] = dia_seg[dia_seg > 0] + offset[slice_label[ii]]
        sys_seg[sys_seg > 0] = sys_seg[sys_seg > 0] + offset[slice_label[ii]]
        
        if slice_label[ii] == 3:
            if (np.sum(dia_seg) == 0) or (np.sum(sys_seg) == 0):
                dia_seg = heart_xy_dia.copy()
                dia_seg[dia_seg==2] = 17
                dia_seg[dia_seg==1] = 0
                dia_seg[dia_seg==3] = 0
                
                sys_seg = heart_xy_sys.copy()
                sys_seg[sys_seg==2] = 17
                sys_seg[sys_seg==1] = 0
                sys_seg[sys_seg==3] = 0
                
   
                
        

        heart_aha17_4d[:, :, ii, dia_frame] = dia_seg
        heart_aha17_4d[:, :, ii, sys_frame] = sys_seg
    
    
    #=====================process basal
    go = True
    slice_label_temp = slice_label.copy()
    while go:
        if np.any(np.diff(slice_label_temp)==-10):
            index11 = int(np.where(np.diff(slice_label_temp)==-10)[0])
            index1 = index11 + 1
            slice_label_temp[index11] = 1
            go = True

        elif np.any(np.diff(slice_label_temp)==10):
            index11 = int(np.where(np.diff(slice_label_temp)==10)[0] + 1)
            index1 = index11 - 1
            slice_label_temp[index11] = 1
            go = True
        else:
            index11 = -1
            go = False

        if go:
            mask11_sys = heart[:, :, index11, sys_frame]
            mask11_dia = heart[:, :, index11, dia_frame]
            seg1_sys = heart_aha17_4d[:, :, index1, sys_frame]
            seg1_dia = heart_aha17_4d[:, :, index1, dia_frame]

            seg11_dia = proc_basal_without_RV(seg1_dia, mask11_dia)
            seg11_sys = proc_basal_without_RV(seg1_sys, mask11_sys)
            heart_aha17_4d[:, :, index11, sys_frame] = seg11_sys
            heart_aha17_4d[:, :, index11, dia_frame] = seg11_dia
            #plt.figure()
            #plt.imshow(seg11)
            #print(slice_label_temp)  


    heart_aha17_4d[heart==1] = 18
    heart_aha17_4d[heart==3] = 19

    if (result_dir is not None) and file_input:        
        result_f = join(result_dir, new_f)        
        nii_label = nibabel.Nifti1Image(heart_aha17_4d.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)

    return heart_aha17_4d
