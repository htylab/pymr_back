# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import copy


def center(LV):
    LV2 = ndi.binary_fill_holes(LV)
    [cx, cy] = ndi.center_of_mass(LV2)
    return [int(cx), int(cy)]
# %%


def circular_sector(r_range, theta_range, LV_center):
    theta = theta_range/180*np.pi
    z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
    xall = -np.imag(z) + LV_center[0]
    yall = np.real(z) + LV_center[1]
    return xall, yall
#


def get_xall_yall(xall, yall, deg, RV_fov):
    if deg == -1:
        m3_xall = np.round(xall.flatten())
        m3_yall = np.round(yall.flatten())
    else:
        m3_xall = np.round(xall[:, deg].flatten())
        m3_yall = np.round(yall[:, deg].flatten())
    mask = (m3_xall >= 0) & (m3_yall >= 0) & \
           (m3_xall < RV_fov.shape[0]) & (m3_yall < RV_fov.shape[1])
    m3_xall = m3_xall[np.nonzero(mask)].astype(int)
    m3_yall = m3_yall[np.nonzero(mask)].astype(int)
    return m3_xall, m3_yall
# %%


def radial_projection(RV, LV_center):
    cx, cy = LV_center
    M = copy.copy(RV)
    sum_curve = np.zeros(360)
    rr = int(max(M.shape))
    xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                 np.arange(0., 360.), LV_center)

    if False:
        M[cx, cy] = 1
        plt.figure()
        for num in range(0, 360, 30):
            m3_xall, m3_yall = get_xall_yall(xall, yall, num, M)
            M[m3_xall, m3_yall] = 1
            plt.imshow(M)

    for num in range(0, 360):
        m3_xall, m3_yall = get_xall_yall(xall, yall, num, M)
        radial = np.array([])
        for nn in range(m3_xall.size):
            radial = np.append(radial, RV[m3_xall[nn], m3_yall[nn]])
        sum_curve[num] = radial.sum()

    return sum_curve
# %%


def MeanROI(LV_seg, cal_map):
    #max_index = int(np.max(LV_seg))
    val_mean = np.zeros(6)
    val_median = np.zeros(6)
    voxels = [np.zeros(1)]*6
    for ii in range(6):
        aha_index = ii + 1
        if (LV_seg == aha_index).sum() > 0:
            voxels[ii] = cal_map[np.nonzero(LV_seg == aha_index)]
            val_mean[ii] = np.mean(voxels[ii]).round(3)
            val_median[ii] = np.median(voxels[ii]).round(3)
    return val_mean, val_median,voxels
# %%


def labelit(angles, LV_wall, LV_center):
    LV_seprt = LV_wall * 0
    angles2 = np.append(angles, angles[0])
    rr = int(max(LV_wall.shape))
    angles2 = np.rad2deg(np.unwrap(np.deg2rad(angles2)))

    for ii in range(angles2.size-1):
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     np.arange(angles2[ii], angles2[ii+1],
                                     0.5), LV_center)
        xall, yall = get_xall_yall(xall, yall, -1, LV_wall)
        LV_seprt[xall, yall] = ii + 1

    return LV_seprt*LV_wall
# %%


def UP_DN(sumar):
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt
    x = np.arange(sumar.size)
    y = sumar
    maxv = np.argmax(y)
    y = medfilt(y)

    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    def fit(x, y):
        p0 = [np.max(y), np.argmax(y)+x[0], 1.]
        try:
            coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
            A, mu, sigma = coeff
        except:
            mu = 0
            sigma = 0
        return mu, sigma

    mu, sigma = fit(x[:maxv], y[:maxv])
    uprank1 = mu - sigma*2.5
    mu, sigma = fit(x[maxv:], y[maxv:])
    downrank1 = mu + sigma*2.5
    if downrank1 == 0:
        downrank1 = 360

    uprank2 = np.nonzero(y > 5)[0][0]
    downrank2 = np.nonzero(y > 5)[0][-1]
    uprank = max(uprank1, uprank2)
    downrank = min(downrank1, downrank2)

    return int(uprank), int(downrank)
#  %%


def degree_calcu(UP, DN, seg_num):
    anglelist = np.zeros(seg_num)
    if seg_num == 4:
        anglelist[0] = DN-180.
        anglelist[1] = UP
        anglelist[2] = DN
        anglelist[3] = UP+180.

    if seg_num == 6:
        anglelist[0] = DN-180.
        anglelist[1] = UP
        anglelist[2] = (UP+DN)/2.
        anglelist[3] = DN
        anglelist[4] = UP+180.
        anglelist[5] = anglelist[2]+180.
    anglelist = (anglelist + 360) % 360

    return anglelist.astype(int)
# %%


def cal_slice(slice_dict, slicename):
    # print('slice name: %s' % slicename)
    if slice_dict is None:
        if slicename == 'A':
            val_mean = np.zeros(4)
            empty_voxel = [np.array([0])]*4
        else:
            val_mean = np.zeros(6)
            empty_voxel = [np.array([0])]*6
        return val_mean,val_mean, -1, -1, None, None, None, None, empty_voxel
    slice_index = {'B': 1, 'M': 2, 'A': 3}
    LV_walls = slice_dict['LV_walls']
    CalMap = slice_dict['map']
    LV_center = center(LV_walls)
    LV_walls[np.isnan(LV_walls)] = 0.
    LV_walls[np.where(LV_walls != 0.)] = 1.

    if ('RV') in slice_dict:
        slice_rv = slice_dict['RV']
    else:
        from RV_mask import find_rv_mask as FRV
        slice_rv = FRV(CalMap, LV_walls, slice_index[slicename])

    sum_curve = radial_projection(slice_rv, LV_center)
    uprank, downrank = UP_DN(sum_curve)
    rr = int(max(CalMap.shape))
    xall, yall = circular_sector(np.arange(0, rr, 2),
                                 np.array([uprank, downrank]), LV_center)
    CalMap2 = copy.copy(CalMap)
    xall, yall = get_xall_yall(xall, yall, -1, slice_rv)
    CalMap2[xall, yall] = 0.

    if slicename == 'A':
        anglelist = degree_calcu(uprank, downrank, 4)
    else:
        anglelist = degree_calcu(uprank, downrank, 6)
    LV_Seg = labelit(anglelist, LV_walls, LV_center)
    val_mean, val_median, voxels = MeanROI(LV_Seg, CalMap)
    AHAlabel_offset={'B': 1, 'M': 7, 'A': 13}
    LV_Seg[LV_Seg>0] += AHAlabel_offset[slicename]

    if slicename == 'A':
        val_mean = val_mean[:4]
        val_median = val_median[:4]
        voxels = voxels[:4]

    return  val_mean, val_median, uprank, downrank, LV_Seg, slice_rv, CalMap2, sum_curve, voxels
# %%
def save_fig(AHAdt,figfile):
    def color_over(t1map,lvseg):
        t1map = t1map/t1map.flatten().max()*255
        t1map = t1map*(lvseg<1)+(lvseg>0)*256+lvseg
        return t1map

    cmap = plt.cm.gray
    jet = plt.cm.nipy_spectral
    cmaplist = [cmap(i) for i in range(cmap.N)] + [ii for ii in jet(np.arange(0,230,230//17))]
    cmap_new = cmap.from_list('Custom cmap', cmaplist)
    ima_all=[]
    for key in ['B','M','A']:
        try:
            t1map = color_over(AHAdt['map_'+key],AHAdt['LV_Seg_'+key])
            ima_all.append(np.hstack([t1map,np.ones((t1map.shape[0],10))*255]))
        except:
            pass
    ima_stack = np.hstack([im for im in ima_all])
    fig = plt.figure(figsize=(30,10), dpi=600)
    plt.imshow(ima_stack,cmap=cmap_new),plt.clim([0, len(cmaplist)])
    plt.axis('off')
    plt.savefig(figfile, bbox_inches='tight')
    plt.close(fig)
#%%

def AHA17(B=None, M=None, A=None,figname=None):
    dcAHA = {'mean17': np.zeros(17), 'median17': np.zeros(17)}
    dcRange = {'B': (0, 6), 'M': (6,12), 'A': (12,16)}
    dcAHA['voxels_all'] = []
    for ii, slicen in enumerate(['B', 'M', 'A']):
        istart, iend = dcRange[slicen]
        dcSlice = eval(slicen)
        dcAHA['mean17'][istart:iend],\
        dcAHA['median17'][istart:iend],\
        dcAHA['UP_' + slicen],\
        dcAHA['DN_' + slicen],\
        dcAHA['LV_Seg_' + slicen],\
        dcAHA['RV_' + slicen],\
        dcAHA['map_angle_' + slicen],\
        dcAHA['radial_' + slicen], \
        voxels  = cal_slice(copy.copy(dcSlice), slicen)
        if dcSlice is not None:
            dcAHA['map_' + slicen] = dcSlice['map']

        dcAHA['voxels_all'] += voxels
            #septum voxels [2,3,8,9,14]
    septum_vox = [dcAHA['voxels_all'][ii-1] for ii in [2,3,8,9,14]]

    septum_vox = np.concatenate(tuple(septum_vox))
    septum_vox = septum_vox[septum_vox>0]
    dcAHA['septum_voxels'] = septum_vox


    dcAHA['septum_median'] = np.round(np.median(septum_vox),3)
    dcAHA['septum_mean'] = np.round(np.mean(septum_vox),3)

    nonsep_vox = [dcAHA['voxels_all'][ii-1] for ii in [1,4,5,6,10,11,12,13,15,16]]
    nonsep_vox = np.concatenate(tuple(nonsep_vox))
    nonsep_vox = nonsep_vox[nonsep_vox>0]
    dcAHA['nonsep_voxels'] = nonsep_vox

    dcAHA['nonsep_median'] = np.round(np.median(nonsep_vox),3)
    dcAHA['nonsep_mean'] = np.round(np.mean(nonsep_vox),3)

    global_vox = dcAHA['voxels_all']
    global_vox = np.concatenate(tuple(global_vox))
    global_vox = global_vox[global_vox>0]
    dcAHA['global_voxels'] = global_vox

    dcAHA['global_median'] = np.round(np.median(global_vox),3)
    dcAHA['global_mean'] = np.round(np.mean(global_vox),3)

    if figname is not None:
        save_fig(dcAHA,figname)


    return dcAHA
