# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage as ndi
import scipy.ndimage.measurements as ndim


def center(LV) :
    [cy,cx]=ndi.center_of_mass(ndi.binary_fill_holes(LV))
    cx=int(cx)
    cy=int(cy)
    return [cx,cy]


"""
----------------------------------------------------------------
sum_ar()
INPUT: m3_sum_num=label 後的mask , A_var=指定那塊面積[-1]為最大
OUTPUT:分割後的面積(array)
-----------------------------------------------------------------
"""

def sum_ar(M,A_var) :

    m3_sum_num=copy.copy(ndim.label(M)[0])
    sv = np.zeros(np.max(m3_sum_num)+1)
    for ii in range(1,np.max(m3_sum_num)+1) :
        sv[ii] = np.sum(m3_sum_num==ii)
    m3_coor_1 = np.sort(sv)[A_var]  #RV 面積由小排到大
    m3_coor_1 = np.argwhere(sv==m3_coor_1)[0,0]
    RV_maxarea = ndim.label(m3_sum_num)[0]==m3_coor_1 #RV

    return RV_maxarea


"""
========================================
Fill_rate()
=======================================
"""
def Fill_rate(mask):
    M=copy.copy(mask)
    Right,Left=[np.max(np.argwhere(M==1)[:,1]),np.min(np.argwhere(M==1)[:,1])]
    High,Low=[np.max(np.argwhere(M==1)[:,0]),np.min(np.argwhere(M==1)[:,0])]
    M_area=np.sum(M==1)
    length=abs(High-Low)
    Weigth=abs(Right-Left)
    Area=np.float(length*Weigth)
    fill_rate=M_area/Area
    return fill_rate


def cir_rang(rr,p1,p2,mask,LV_center) :

    theta=np.append(p1,p2)
    M=copy.copy(mask)

    cx, cy = LV_center

    for ii in range(2) :
        m3_xall=np.array([])
        m3_yall=np.array([])
        if ii == 0:
            m3_theta=np.arange(0,theta[ii])/180.*np.pi
        elif ii ==1:
            m3_theta=np.arange(theta[ii],360.)/180.*np.pi

        for r in np.arange(0,rr,0.5) :
                z=r*np.exp(1.0j*m3_theta)
                x=-np.imag(z)
                y=np.real(z)
                m3_xall=np.append(m3_xall,x)
                m3_yall=np.append(m3_yall,y)
                #plt.figure(4),plt.plot(m3_yall,m3_xall)

        m3_xall2=m3_xall.astype(int)+int(cy)
        m3_yall2=m3_yall.astype(int)+int(cx)

        M[m3_xall2,m3_yall2]=0

    M=sum_ar(M,-1)

    return M










"""
==============================================================
找出RV MASK
find_rv_mask():
INPUT:T1map,lvwall,qq=slice(1,2,3)
OUTPUT:RV_MASK
===============================================================
"""


def find_rv_mask(T1map,lv_wall,qq) :
    global m3_coor_1

    m3_T1map=copy.copy(T1map)

    lvwall=copy.copy(lv_wall)

    lvwall[np.isnan(lvwall)] = 0.

    lvwall[np.where(lvwall!=0)] = 1.

    [m3_x,m3_y] = center(lvwall)

    m3_lvwall=copy.copy(lvwall)

    m3_new=np.zeros((m3_T1map.shape[0],m3_T1map.shape[1]))

    """
    ====================================================
    draw_cir()
    INPUT:qq=1,2,3,4 m3_r=半徑 m3_theta=畫幾度的圓
    OUTPUT:[x,y]圓的級座標
    ====================================================
    """

    def draw_cir(qq,m3_r,m3_theta) :

        m3_xall=np.array([])

        m3_yall=np.array([])

        if qq==3 and m3_r==0 and m3_theta==0 :

            m3_r=15

            m3_theta=np.arange(0.,359.)/180.*np.pi

        elif qq==1 and m3_r==0 and m3_theta==0 or qq==2 and m3_r==0 and m3_theta==0 :

            m3_theta=np.arange(0.,359.)/180.*np.pi

            m3_r=35

        else :

            m3_r=m3_r

            m3_theta=m3_theta

        for r in np.arange(0,m3_r,0.5) :

            z=r*np.exp(1.0j*m3_theta)

            x=-np.imag(z)

            y=np.real(z)

            m3_xall=np.append(m3_xall,x)

            m3_yall=np.append(m3_yall,y)

        #plt.figure(4),plt.plot(yall,xall)

        m3_xall2=m3_xall.astype(int)+int(m3_y)

        m3_yall2=m3_yall.astype(int)+int(m3_x)

        return [m3_xall2,m3_yall2]



    """
    --------
    確認心臟位置
    ----
    """

    struc=[[0,1,0],
           [1,1,1],
           [0,1,0]]

    m3_new[np.argwhere(m3_T1map==0)[:,0],np.argwhere(m3_T1map==0)[:,1]] = 0.

    m3_lv = copy.copy(ndi.binary_fill_holes(m3_lvwall))

    m3_lv_coor = ndim.label(m3_lv,structure=struc)[0][m3_y,m3_x]

    m3_lv = ndim.label(m3_lv,structure=struc)[0]==m3_lv_coor

    #plt.figure(5).suptitle('LV', fontsize=14, fontweight='bold'),plt.imshow(m3_lv,cmap='gray')

    m3_bld = copy.copy(np.logical_xor(m3_lv,m3_lvwall))

    m3_bld=sum_ar(m3_bld,A_var=-1)

    #plt.figure(6).suptitle('LV_blood', fontsize=14, fontweight='bold'),plt.imshow(m3_bld,cmap='gray')

    sort=np.sort(m3_T1map[np.argwhere(m3_bld==1)[:,0],np.argwhere(m3_bld==1)[:,1]])

    m3_T1map_max=sort[np.argwhere(sort < 1.9)[-1]]

    m3_T1map_min=sort[np.argwhere(sort > 1.3)[0]]

    #print max_sort,min_sort

    for jj in range(m3_T1map.shape[0]) :

        for kk in range(m3_T1map.shape[1]) :

            if m3_T1map[jj,kk] > m3_T1map_max :

                m3_new[jj,kk] = 0

            elif m3_T1map[jj,kk] < m3_T1map_min :

                m3_new[jj,kk] = 0

            else :

                m3_new[jj,kk] = 1



    m3_new[np.argwhere(m3_lv==1)[:,0],np.argwhere(m3_lv==1)[:,1]] = 1.

    m3_hrt = copy.copy(ndi.binary_fill_holes(m3_new))

    #plt.figure(7).suptitle('Heart mask', fontsize=14, fontweight='bold'),plt.imshow(m3_hrt,cmap='gray')

    m3_coor = ndim.label(m3_hrt,structure=struc)[0][m3_y,m3_x]

    m3_new = ndim.label(m3_hrt,structure=struc)[0]==m3_coor

    hrt_label = copy.copy(m3_new)

    #plt.figure(8).suptitle('Heart mask label', fontsize=14, fontweight='bold'),plt.imshow(hrt_label,cmap='gray')

    m3_lv_Delate=copy.copy(ndi.binary_dilation(m3_lv))#**Delate

    m3_new[np.argwhere(m3_lv==1)[:,0],np.argwhere(m3_lv==1)[:,1]] = 0.#**Delate

    #m3_new = ndi.binary_erosion(m3_new)

    #m3_new = sum_ar(m3_new,-1)

    m3_new = ndi.binary_fill_holes(m3_new)

    RV_labl_b = copy.copy(m3_new)

    #plt.figure(9).suptitle('RV Mask before label', fontsize=14, fontweight='bold'),plt.imshow(RV_labl_b,cmap='gray')

    """
    ---------
    找RV MASK
    ---------
    """

    m3_sum_var = ndim.label(m3_new,structure=struc)[0]

    '''
    =============================
    找出RV與LV連接點最多的MASK
    ============================
    '''

    def conect_are(RV) :

        LV_rang = np.zeros(360)

        RV_rang = np.zeros(360)

        #m3_heart = np.logical_or(RV,m3_lv)

        m3_heart = np.logical_or(RV,m3_lv)#**Delate

        conet_area = 0

        for rag in range(30,285) :

            [lv_x,lv_y] = draw_cir(0,45,np.arange(rag,rag+1))

            LV_rang[rag] = np.argwhere(m3_lv[lv_x,lv_y]==0)[0]

            if np.any(m3_heart[lv_x,lv_y]==0) :

                RV_rang[rag] = np.argwhere(m3_heart[lv_x,lv_y]==0)[0]

                #print rag

            else :

                RV_rang[rag] = np.argwhere(m3_heart[lv_x,lv_y]!=0)[-1]

            if abs(LV_rang[rag] - RV_rang[rag]) != 0 :

                conet_area = conet_area +1

            #print rag, LV_rang[rag],RV_rang[rag]

        return conet_area

    """
    ---------------------------------
    找RV Mask面積最大 且 與LV連接點最多
    ---------------------------------
    """

    stat = 1

    are_var = -1

    conect_num = 0

    labl_num_1 = np.zeros(np.max(m3_sum_var))

    if np.max(m3_sum_var) < 4 :

       ln=np.max(m3_sum_var)+1

    else :

       ln = 4

    max_are = 0 #conect point

    #max_sum = 0 #area

    while stat==1 :

        for are_var in range(-1,-ln,-1) :

            m3_new = sum_ar(m3_sum_var,A_var=are_var)

            m3_new_area = np.sum(m3_new==1)

            #print m3_new_area

            if ndim.label(np.logical_or(m3_new,m3_lv),structure=struc)[1] == 1 and m3_new_area > 100. :#有連接且面積大於100

                #plt.imshow(ndim.label(np.logical_or(m3_new,m3_lv),structure=struc)[0])

                #print are_var

                labl_num_1[conect_num] = are_var

                conect_num = conect_num+1

                #print 'conect:',conect_are(m3_new)

                #if conect_are(m3_new) > max_are and (np.sum(m3_new==1) > max_sum) :

                if conect_are(m3_new) > max_are  :

                    if abs(conect_are(m3_new) - max_are) and max_are != 0 < 25 :

                        RV_max_num = are_var+1

                    else :
                        max_are = conect_are(m3_new)

                        #max_sum = np.sum(m3_new==1)

                        RV_max_num = are_var


        m3_new = sum_ar(m3_sum_var,A_var=RV_max_num)

        RV_M = copy.copy(m3_new)

       # plt.figure(10).suptitle('RV Mask after label', fontsize=14, fontweight='bold'),plt.imshow(RV_M,cmap='gray')

        #,structure=struc
        if ndim.label(np.logical_or(m3_new,m3_lv),structure=struc)[1] != 1 : #最大面積RV與LV 沒有連在一起*

            print (ndim.label(np.logical_or(m3_new,m3_lv),structure=struc)[1])

            #plt.figure(499),plt.imshow(ndim.label(np.logical_or(m3_new,m3_lv),structure=struc)[0])

            stat=1

            are_var=are_var-1

            if abs(are_var)>np.max(m3_sum_var) :

                stat=0

                m3_new=np.zeros((m3_T1map.shape[0],m3_T1map.shape[1]))

                print ("can not find RV MASK")
        else:

            stat=0

    [m3_new,m3_new_num] = ndim.label(m3_new)

    print ("RV_area=%d"%(np.sum(m3_new==1)))


    fil_rate_org=Fill_rate(m3_new)

    print ('extent',fil_rate_org)



    if np.sum(m3_new)>1000 :

        m3_new=ndi.binary_erosion(m3_new)

        fil_rate_org=Fill_rate(m3_new)

        if fil_rate_org < 0.3  :

            m3_new=ndi.binary_erosion(m3_new)

        else :

            m3_new=ndi.binary_erosion(m3_new)





    m3_new = ndim.label(m3_new)[0]

    m3_new = sum_ar(m3_new,A_var=-1)

    '''
    if qq==1:

        plt.figure(51).suptitle('RV_B', fontsize=14, fontweight='bold'),plt.imshow(m3_new,cmap='gray')

    elif qq==2:

        plt.figure(52).suptitle('RV_M', fontsize=14, fontweight='bold'),plt.imshow(m3_new,cmap='gray')

    else :

        plt.figure(53).suptitle('RV_A', fontsize=14, fontweight='bold'),plt.imshow(m3_new,cmap='gray')


    '''
    return m3_new
