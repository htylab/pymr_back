# -*- coding: utf-8 -*-
'''
example usage:
%matplotlib inline
import matplotlib.pyplot as plt
from pymr.heart import ahaseg
heart_mask = (LVbmask, LVwmask, RVbmask)
label_mask = ahaseg.get_seg(heart_mask, nseg=4)
plt.imshow(label_mask)
'''
import numpy as np
from scipy import ndimage

def get_heartmask(heart_mask):
    if isinstance(heart_mask, tuple):
        #backward compatibility
        LVbmask, LVwmask, RVbmask = heart_mask
    else:
        LVbmask = (heart_mask==1)
        LVwmask = (heart_mask==2)
        RVbmask = (heart_mask==3)
    return LVbmask, LVwmask, RVbmask

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

def circular_sector(r_range, theta_range, LV_center):
    cx, cy = LV_center
    theta = theta_range/180*np.pi
    z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
    xall = -np.imag(z) + cx
    yall = np.real(z) + cy
    return xall, yall

def get_theta(sweep360):
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt
    sumar = np.array(sweep360)
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

def get_sweep360(LVwmask, RVbmask):

    LV_center = ndimage.center_of_mass(LVwmask)
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)

    sweep360 = []
    for theta in range(360):
        #print(theta)
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     theta, LV_center)
        projection = ndimage.map_coordinates(RVbmask, [xall, yall], order=0).sum()
        sweep360.append(projection)
    return np.array(sweep360)


def get_angle(heart_mask, nseg=4):
    def sector_mask(xall, yall, LVwmask):
        smask = LVwmask * 0
        xall = np.round(xall.flatten())
        yall = np.round(yall.flatten())

        mask = (xall >= 0) & (yall >= 0) & \
               (xall < LVwmask.shape[0]) & (yall < LVwmask.shape[1])
        xall = xall[np.nonzero(mask)].astype(int)
        yall = yall[np.nonzero(mask)].astype(int)
        smask[xall, yall] = 1
        return smask
    
    #step 1: sweep 360 and get AHA angles
    
    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask)
    #import time
    #t = time.time()
    sweep360 = get_sweep360(LVwmask, RVbmask)
    UP, DN = get_theta(sweep360)
    anglelist = degree_calcu(UP, DN, nseg)
    
    #step 2: calculate mask360, 360 points with AHA labels

    angles2 = np.append(anglelist, anglelist[0])
    angles2 = np.rad2deg(np.unwrap(np.deg2rad(angles2)))
    mask360 = np.zeros((360, ))
    for ii in range(angles2.size-1):
        temp = np.arange(angles2[ii], angles2[ii + 1]).astype(np.int)
        temp[temp >= 360] = temp[temp >= 360] - 360
        mask360[temp] = ii + 1
        
    #step 3: calculating AHA sector
    LV_center = ndimage.center_of_mass(LVwmask)
    AHA_sector = LVwmask * 0
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)


    for ii in range(angles2.size-1):
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     np.arange(angles2[ii], angles2[ii+1],
                                     0.5), LV_center)
        smask = sector_mask(xall, yall, LVwmask)
        
        AHA_sector[smask > 0] = (ii + 1)
        
    
    return anglelist, mask360, AHA_sector
#print(time.time() - t)
#plt.figure()
#plt.imshow(label_mask)

def get_seg(heart_mask, nseg=4):
    LVbmask, LVwmask, RVbmask = get_heartmask(heart_mask)

    _, _, AHA_sector = get_angle(heart_mask, nseg)
    #label_mask = labelit(anglelist, LVwmask)
    label_mask = AHA_sector * LVwmask

    return label_mask
