# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:45:41 2023

@author: william
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap



# read nc data
MSLP = nc.Dataset("./data/ERA5_MSL_2017_12_21.nc")#
Q = nc.Dataset("./data/ERA5_Q_2017_12_21.nc")
SP = nc.Dataset("./data/ERA5_SP_2017_12_21.nc")
T = nc.Dataset("./data/ERA5_T_2017_12_21.nc")#
U = nc.Dataset("./data/ERA5_U_2017_12_21.nc")#
V = nc.Dataset("./data/ERA5_V_2017_12_21.nc")#
W = nc.Dataset("./data/ERA5_W_2017_12_21.nc")
Z = nc.Dataset("./data/ERA5_Z_2017_12_21.nc")#
prec = nc.Dataset("./data/GPM.daily.Asia.2017.12.21.nc")#
# print(Q)

# read variables
lat = MSLP.variables['lat'][:]          # -10~55  -> 261
lon = MSLP.variables['lon'][:]          #  90~160 -> 281
psl = MSLP.variables['psl'][:]/100      # pressure level
ta = T.variables['ta'][:]               # temp
plev = T.variables['plev'][:]/100       # pressure level
ua = U.variables['ua'][:]               # u wind
va = V.variables['va'][:]               # v wind
preci = prec.variables['preci'][:]      # precipitation
pre_lat = prec.variables['latitude'][:] # prec lat
pre_lon = prec.variables['longitude'][:]# prec lon
geop = Z.variables['geopotential'][:]   # geopotential (phi)
hus = Q.variables['hus'][:]             # qv
sp = SP.variables['sp'][:]/100          # surface pressure (filter the data not real)
w_NON_CDM = W.variables['w_NON_CDM'][:] # verticle wind?


'''plot'''
# fig = plt.figure(figsize=(8, 12))

# setup basemap
m = Basemap(projection = 'cea', llcrnrlat=-10, urcrnrlat=55, llcrnrlon=90,
            urcrnrlon=160, resolution='c')
# draw coastlines  
m.drawcoastlines()
m.drawparallels(np.arange(-10, 61, 10), labels=[1,0,0,0], color='#d3d3d3', zorder = 2)
m.drawmeridians(np.arange(90,161,10),labels=[0,0,0,1], color='#d3d3d3', zorder = 2)

# projection --> map lon, lat 
lon2,lat2=np.meshgrid(lon,lat)
xx, yy = m(lon2, lat2)
# for precipitation projection
pre_lon2,pre_lat2=np.meshgrid(pre_lon,pre_lat)
pre_x, pre_y = m(pre_lon2, pre_lat2)

sp = sp[0,:,:]
flip_sp = np.flip(sp, axis=0)
'''mask'''      # surface pressure mask out < 1000
sp_mask_1000 = flip_sp > 1000
real_ua_1000 = np.where(sp_mask_1000,ua[0,0,:,:],np.nan)
real_va_1000 = np.where(sp_mask_1000,va[0,0,:,:],np.nan)

'''winds barb'''        # flip the y axes
ua_1000mb = real_ua_1000
va_1000mb = real_va_1000
step = 10
for i in range(0, len(lat), step):
    for j in range(0, len(lon), step):
        if i>261 or j>281:
            pass
        else:
            m.barbs(xx[i,j], yy[i,j], ua_1000mb[i,j], va_1000mb[i,j], length=3.7, 
                sizes=dict(emptybarb=0.1, spacing=0.1),pivot='middle',color='#c0c0c0', 
                zorder = 3)

'''isobar contour'''
p_thin = ([984, 992, 1000, 1008, 1016, 1024])   # thin
p_thick = np.arange(996, 1028, 8)               # thick
psl_2D = psl[0,:,:]                             # get level psl
# plt.grid(axis="both", linestyle="dotted", color="b")
thin = m.contour(xx, yy, psl_2D, levels=p_thin, linewidths=.8, colors='k', zorder=4)
thick = m.contour(xx, yy, psl_2D, levels=p_thick, linewidths=1.2, colors='k', zorder=4)
plt.clabel(thick,fontsize=7.5, inline=True, fmt='%1.0f')

# qq = plt.quiver(lon,lat,ua_1000mb,va_1000mb,M)
# plt.quiver(lon, lat, ua_1000mb, va_1000mb, angles='xy', scale_units='xy', scale=0.1)
# M = (ua_1000mb**2 + va_1000mb**2)**0.5

'''color map'''
list_cmap = ListedColormap(['#a0fffa','#00cdff','#0096ff','#0069ff',
                        '#329600','#32ff00','#ffff00','#ffc800',
                        '#ff9600','#ff0000','#c80000','#a00000',
                        '#96009b','#c800d2','#ff00f5','#ffc8ff'])
bounds = [1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300]
norm= mpl.colors.BoundaryNorm(bounds, list_cmap.N, extend='max')

'''precipitation'''
prec_2D = preci[0,:,:]                          # get level prec
precipitation = m.contourf(pre_x, pre_y, prec_2D, levels=bounds, cmap=list_cmap,
                    extend='max', norm=norm, zorder=1.5)
cbar = plt.colorbar(precipitation,orientation='vertical', ticks=bounds)
cbar.ax.tick_params(labelsize=8)

'''
extend for sharp top or bottom, norm for boundary and ticks, 
levels for color distribution
'''

'''
pivot for wind characteristic, zorder for cover other variables
'''
# setting
plt.xlabel('Longtitude [\N{DEGREE SIGN}E]', labelpad=20, fontsize=8)
plt.ylabel('Latitude [\N{DEGREE SIGN}N]', labelpad=30, fontsize=8)
# title trouble
title_1 = """2017.12.21 ERA5 reanalysis: 
          \n1000hPa Wind [kt] (gray barbs), MSLP [hPa] (black contours),
          \nGPM: precipitation [mm/day] (color shading)"""
plt.title(title_1, loc='left', fontsize=10)

# plt.savefig("MSLP & precipitation.png", dpi=250)
plt.show()
plt.close()

'''--------------------------------------------------------------------------------'''

# fig = plt.figure(figsize=(5, 5))

# basemap
m = Basemap(projection = 'cea', llcrnrlat=-10, urcrnrlat=55, llcrnrlon=90,
            urcrnrlon=160, resolution='c')
m.drawcoastlines()
m.drawparallels(np.arange(0, 61, 10), labels=[1,0,0,0], color='#d3d3d3', zorder = 2)
m.drawmeridians(np.arange(90,161,10),labels=[0,0,0,1], color='#d3d3d3', zorder = 2)

# geopotential
z_925  = geop[0,1,:,:]/9.8
z_1000 = geop[0,0,:,:]/9.8
z_500  = geop[0,7,:,:]/9.8

'''mask'''      # surface pressure mask out < 925
sp_mask_925 = flip_sp > 925
real_z_925 = np.where(sp_mask_925,z_925,np.nan)

sp_mask_1000 = flip_sp > 1000
real_z_1000 = np.where(sp_mask_1000,z_1000,np.nan)

'''winds barb'''        
real_ua_925 = np.where(sp_mask_925,ua[0,1,:,:],np.nan)
real_va_925 = np.where(sp_mask_925,va[0,1,:,:],np.nan)

step = 10
for i in range(0, len(lat), step):
    for j in range(0, len(lon), step):
        if i>261 or j>281:
            pass
        else:
            m.barbs(xx[i,j], yy[i,j], real_ua_925[i,j], real_va_925[i,j], length=3.7, 
                sizes=dict(emptybarb=0.1, spacing=0.1),pivot='middle', 
                color='#4682b4', zorder = 3)

# isobar
z_thick = np.arange(630, 990, 60)   # thick
z_thin  = np.arange(660, 960, 60)   # thin

thin = m.contour(xx, yy, real_z_925, levels=z_thin, linewidths=.8, colors='k', zorder=4)
thick = m.contour(xx, yy, real_z_925, levels=z_thick, linewidths=1.2, colors='k', zorder=4)
plt.clabel(thick,fontsize=7.5, inline=True, fmt='%1.0f')

# thickness
thickness = z_500 - real_z_1000
thick_ticks = np.arange(5040, 5880, 60)   # ticks
thicknes_plot = m.contour(xx, yy, thickness, levels=thick_ticks, linewidths=1, colors='darkmagenta', zorder=3)
plt.clabel(thicknes_plot,fontsize=7.5, inline=True, fmt='%1.0f')



# PW
hus_1000 = hus[0,0,:,:]
hus_100  = hus[0,11,:,:]
real_hus_1000 = np.where(sp_mask_1000,hus_1000,np.nan)
PW = -(hus_100*100 - real_hus_1000*1000)/9.8
value = PW

'''color map'''
list_cmap = ListedColormap(['#FFEE99','#FFCC65','#FF9932','#F5691C', 
                            '#FC3D3D','#D60C1E'])
bounds = [40,50,60,70,80,90]
norm= mpl.colors.BoundaryNorm(bounds, list_cmap.N, extend='max')

'''PW'''
def qv_area(qv_bottom, qv_top, p_bottom, p_top):
    area = (qv_bottom + qv_top)*(p_bottom - p_top)/2
    return area
area = qv_area(real_hus_1000, hus[0,1,:,:], plev[0], plev[1])
for i in range(len(plev)-2):
    area += qv_area(hus[0,i+1,:,:], hus[0,i+2,:,:], plev[i+1], plev[i+2])

PW = area/9.8*100
value = PW
PW_plot = m.contourf(xx, yy, PW, levels=bounds, cmap=list_cmap,
                    extend='max', norm=norm, zorder=1.5)
cbar = plt.colorbar(PW_plot,orientation='vertical', ticks=bounds)
cbar.ax.tick_params(labelsize=8)


plt.xlabel('Longtitude [\N{DEGREE SIGN}E]', labelpad=20, fontsize=8)
plt.ylabel('Latitude [\N{DEGREE SIGN}N]', labelpad=30, fontsize=8)
title_1 = """2017.12.21 ERA5 reanalysis: PW [kg/m$^{2}$] (color shading), 
          \n925hPa Wind [kt] (blue barbs), 925hPa Z [m] (black contours), 
          \n1000-500hPa Depth [m] (purple contours)"""
plt.title(title_1, loc='left', fontsize=10)
# plt.savefig("Precipitable Water and Thickness.png", dpi=250)
plt.show()
plt.close()

'''--------------------------------------------------------------------------'''
# =============================================================================
# fig = plt.figure(figsize=(6, 6))
# plt.rcParams["figure.figsize"] = (8, 6)
# =============================================================================



# basemap
m = Basemap(projection = 'cea', llcrnrlat=-10, urcrnrlat=55, llcrnrlon=90,
            urcrnrlon=160, resolution='c')
m.drawcoastlines()
m.drawparallels(np.arange(0, 61, 10), labels=[1,0,0,0], color='#d3d3d3', zorder = 2)
m.drawmeridians(np.arange(90,161,10),labels=[0,0,0,1], color='#d3d3d3', zorder = 2)

# mask
sp_mask_850 = flip_sp > 850

'''winds barb'''        
real_ua_850 = np.where(sp_mask_850,ua[0,3,:,:],np.nan)
real_va_850 = np.where(sp_mask_850,va[0,3,:,:],np.nan)

step = 10
for i in range(0, len(lat), step):
    for j in range(0, len(lon), step):
        if i>261 or j>281:
            pass
        else:
            m.barbs(xx[i,j], yy[i,j], real_ua_850[i,j], real_va_850[i,j], length=3.7, 
                sizes=dict(emptybarb=0.1, spacing=0.1),pivot='middle', 
                color='#4682b4', zorder = 3)

'''contour'''
z_850  = geop[0,3,:,:]/9.8
real_z_850 = np.where(sp_mask_850,z_850,np.nan)

z_thick_850 = np.arange(1200, 1620, 60)   # thick
z_thin_850  = np.arange(1230, 1650, 60)   # thin

thin = m.contour(xx, yy, real_z_850, levels=z_thin_850, linewidths=.8, colors='k', zorder=4)
thick = m.contour(xx, yy, real_z_850, levels=z_thick_850, linewidths=1.2, colors='k', zorder=4)
plt.clabel(thick,fontsize=7.5, inline=True, fmt='%1.0f')

'''equivalent potential temperature'''   # --> 850hPa
Rd = 287
Cp = 1004
Lv = 2.5*10**6

def mixing_ratio(qv):
    r = qv/(1-qv)
    return r

def theta(T, P):
    theta = T*(1000/P)**(Rd/Cp)
    return theta
# T_level*(1000/P_level)**((Rd/Cp)*np.exp(Lv*q/(Cp*T_level)))
# theta_e = (T_850 + Lv*r/Cp)*(1000/P_850)**(Rd/Cp)
def EPT(T_level, r, P_level):
    theta_e = (T_level + Lv*r/Cp)*(1000/P_level)**(Rd/Cp)
    return theta_e

real_hus_850 = np.where(sp_mask_850,hus[0,3,:,:],np.nan)
t_850 = ta[0,3,:,:]
r_850 = mixing_ratio(real_hus_850)
theta_e_850 = EPT(t_850, r_850, 850)

'''color map'''
list_cmap = ListedColormap(['#ADFF2F','#FFFF00', '#FFA500', '#FF4500', 
                            '#FF0000'])
bounds = [339,342,345,348,351]
norm= mpl.colors.BoundaryNorm(bounds, list_cmap.N, extend='max')

EPT_plot = m.contourf(xx, yy, theta_e_850, levels=bounds, cmap=list_cmap,
                    extend='max', norm=norm, zorder=1.5)
cbar = plt.colorbar(EPT_plot,orientation='vertical', ticks=bounds)
cbar.ax.tick_params(labelsize=8)


plt.xlabel('Longtitude [\N{DEGREE SIGN}E]', labelpad=20, fontsize=8)
plt.ylabel('Latitude [\N{DEGREE SIGN}N]', labelpad=30, fontsize=8)


title_1 = """2017.12.21 ERA5 reanalysis: 
             \n850hPa \u03B8$_{e}$ [K] (color shading), 850hPa Wind [kt] (blue barbs), 
             \n850 hPa Z [m] (black contours)"""
plt.title(title_1, loc='left', fontsize=10)


# plt.savefig("Equivalent Potential Temperature.png", dpi=250)
plt.show()
plt.close()




