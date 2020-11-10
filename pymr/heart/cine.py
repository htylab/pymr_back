import nibabel
import numpy as np
from ..heart import ahaseg
import matplotlib.pyplot as plt
from os.path import join, basename

def get_slice_label(heart4d):
    heart = heart4d.copy()
    curve3 = np.sum(heart==3, axis=(0, 1, 3))
    curve2 = np.sum(heart==2, axis=(0, 1, 3))
    curve1 = np.sum(heart==1, axis=(0, 1, 3))

    curve_and = (curve3 > 0) & (curve2 > 0) & (curve1 > 0)
    curve_or = (curve3 > 0) | (curve2 > 0) | (curve1 > 0)
    curve_and = curve_and.astype(np.int)
    curve_or = curve_or.astype(np.int)
    curve_diff = curve_or-curve_and
    slice_label = curve_and * 0
    curve_apex = curve_diff.copy()
    curve_apex[:curve_apex.size//2] = 0
    curve_basal = curve_diff.copy()
    curve_basal[curve_apex.size//2:] = 0
    #curve_apex = curve_apex * 4
    
    loc = dict()
    if np.sum(curve_apex) > 0:
        #got_apex = True    
        loc[3] = [1]*1 + [2]*1 + [3]*1
        loc[4] = [1]*1 + [2]*2 + [3]*1
        loc[5] = [1]*2 + [2]*2 + [3]*1
        loc[6] = [1]*2 + [2]*2 + [3]*2
        loc[7] = [1]*2 + [2]*3 + [3]*2
        loc[8] = [1]*3 + [2]*3 + [3]*2
        loc[9] = [1]*3 + [2]*3 + [3]*3
        loc[10] = [1]*3 + [2]*4 + [3]*3
        loc[11] = [1]*4 + [2]*4 + [3]*3
        loc[12] = [1]*4 + [2]*4 + [3]*4
        loc[13] = [1]*4 + [2]*5 + [3]*4
        loc[14] = [1]*5 + [2]*5 + [3]*4
        loc[15] = [1]*5 + [2]*5 + [3]*5
        loc[16] = [1]*5 + [2]*6 + [3]*5
        loc[17] = [1]*6 + [2]*6 + [3]*5
        loc[18] = [1]*6 + [2]*6 + [3]*6
        loc[19] = [1]*6 + [2]*7 + [3]*6
        loc[20] = [1]*7 + [2]*7 + [3]*6
    else:
        loc[3] = [1]*1 + [2]*1 + [3]*1
        loc[4] = [1]*1 + [2]*1 + [3]*1 + [4]*1
        loc[5] = [1]*1 + [2]*2 + [3]*1 + [4]*1
        loc[6] = [1]*2 + [2]*2 + [3]*1 + [4]*1
        loc[7] = [1]*2 + [2]*2 + [3]*2 + [4]*1
        loc[8] = [1]*2 + [2]*3 + [3]*2 + [4]*1
        loc[9] = [1]*3 + [2]*3 + [3]*2 + [4]*1
        loc[10] = [1]*3 + [2]*3 + [3]*3 + [4]*1
        loc[11] = [1]*3 + [2]*3 + [3]*3 + [4]*2
        loc[12] = [1]*3 + [2]*4 + [3]*3 + [4]*2
        loc[13] = [1]*4 + [2]*4 + [3]*3 + [4]*2
        loc[14] = [1]*4 + [2]*4 + [3]*4 + [4]*2
        loc[15] = [1]*4 + [2]*4 + [3]*4 + [4]*3
        loc[16] = [1]*4 + [2]*5 + [3]*4 + [4]*3
        loc[17] = [1]*5 + [2]*5 + [3]*4 + [4]*3
        loc[18] = [1]*5 + [2]*5 + [3]*5 + [4]*3
        loc[19] = [1]*5 + [2]*6 + [3]*5 + [4]*3
        loc[20] = [1]*6 + [2]*6 + [3]*5 + [4]*3
        
    slice_label[curve_and > 0] = loc[np.sum(curve_and)]
    slice_label[curve_apex > 0] = 4
    slice_label[curve_basal > 0] = 5
    return slice_label

def convert(f, result_dir=None):

    if isinstance(f, str):
        temp = nibabel.load(f)
        heart = temp.get_fdata()
        affine = temp.affine
        file_input = True
    else:
        heart = f.copy()
        file_input = False
    reverse = np.median(np.diff(np.sum(heart, axis=(0,1,3)))) > 0
    if reverse:
        heart = heart[:,:, ::-1, :]    
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
        if (slice_label[ii] == 0) or (slice_label[ii] == 5):
            continue

        if slice_label[ii] == 4:
            temp = heart[:, :, ii, :].copy()
            temp[temp==2] = 17
            temp[temp==1] = 0
            temp[temp==3] = 0
            heart_aha17_4d[:, :, ii, :] = temp
            continue


        heart_xyt = heart[:, :, ii, :].copy()

        curve = np.sum(heart_xyt, axis=(0, 1))
        dia_frame = np.argmax(curve)
        curve[curve==0] = 1e20
        sys_frame = np.argmin(curve)
        #print(dia_frame, sys_frame)
        heart_xy_dia = heart_xyt[..., dia_frame]
        heart_xy_sys = heart_xyt[..., sys_frame]
        if slice_label[ii] == 3:
            nseg = 4
        else:
            nseg = 6

        if (np.sum(heart_xy_dia==3) < 5) or (np.sum(heart_xy_sys==3) < 5):
            slice_label[ii] = 5
            continue


        dia_seg = ahaseg.get_seg((heart_xy_dia==1, heart_xy_dia==2, heart_xy_dia==3), nseg)
        sys_seg = ahaseg.get_seg((heart_xy_sys==1, heart_xy_sys==2, heart_xy_sys==3), nseg)

        dia_seg[dia_seg > 0] = dia_seg[dia_seg > 0] + offset[slice_label[ii]]
        sys_seg[sys_seg > 0] = sys_seg[sys_seg > 0] + offset[slice_label[ii]]

        heart_aha17_4d[:, :, ii, dia_frame] = dia_seg
        heart_aha17_4d[:, :, ii, sys_frame] = sys_seg

    #print('reverse:', reverse)
    if reverse:
        heart_aha17_4d = heart_aha17_4d[:,:, ::-1, :]

    if (result_dir is not None) and file_input:        
        result_f = join(result_dir, basename(f))        
        nii_label = nibabel.Nifti1Image(heart_aha17_4d.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)
    
    return heart_aha17_4d, sys_frame, dia_frame

import matplotlib.pyplot as plt
def montage(X, cmap='viridis'):
    m, n, count = X.shape    
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count: 
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = X[:, :, image_id]
            image_id += 1
                    
    #pylab.imshow(flipud(rot90(M)), cmap=colormap)
    #pylab.axis('off')
    plt.figure(figsize=(10, 10))
    plt.imshow(M, cmap=cmap)
    plt.axis('off')          
    return M

    

