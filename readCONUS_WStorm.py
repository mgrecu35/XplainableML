import numpy as np
from wrf import *
import wrf as wrf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# draw tissot's indicatrix to show distortion.
fname='wrfout_d04_2018-06-01_09:00:00'
#fname='/media/grecu/ExtraDrive1/SOceansPF/wrfout_d03_2018-08-08_09:00:00'
fname='/media/grecu/ExtraDrive1/conusCold/wrfout_d03_2018-01-12_11:00:00'
ncFile=Dataset(fname)
from matplotlib import pyplot as plt

p = wrf.getvar(ncFile, "pressure")
th = wrf.getvar(ncFile, "th")

th_850 = wrf.interplevel(th, p, 850.0)

tsk=ncFile['TSK'][-1,:,:]
lon=ncFile['XLONG'][-1,:,:]
lat=ncFile['XLAT'][-1,:,:]
th_tsk=tsk*(p[0,:,:]/1000.)**0.287
diff=th_850-tsk

from wrf_io import read_wrf
import combAlg as cAlg
qr,qs,qg,ncr,ncs,ncg,qc,rho,z,T,tsk,w_10,qv,press,z=read_wrf(fname,0)

#tb = sdsu.radtran(umu,nlyr,sfcTemp,tLayer,height[:,ny,i],\
#                         kext1d[:,ifreq],salb1d[:,ifreq],asym1d[:,ifreq],\
#                         fisot,emis,ebar)
#abair,abswv = sdsu.gasabsr98(freq,t2[k,ny,i],qv[k,ny,i],\
#                                             press[k,ny,i])
#z_clw=sdsu.gcloud(freq,t2[k,ny,i],qc[k,ny,i]*1e3)
dz=z[1:,:,:]-z[:-1,:,:]
swp=((qs+qg)*rho*dz).sum(axis=0)*1e3
rwp=((qr)*rho*dz).sum(axis=0)*1e3
cwp=((qc)*rho*dz).sum(axis=0)*1e3
a=np.nonzero(swp>0.5)
plt.subplot(211)
plt.hist(np.log10(swp[a]),bins=-3+np.arange(12)*0.5)
plt.subplot(212)
plt.hist(np.log10(cwp[a]),bins=-3+np.arange(12)*0.5)

swc=rho*(qs)*1e3
gwc=rho*qg*1e3
rwc=rho*qr*1e3
ncr=ncr*rho*1
ncg=ncg*rho*1
ncs=(ncs)*rho*1
from scattering import *
from scipy.special import gamma as gam

calcScatt(rwc,swc,gwc,ncr,ncs,ncg,scatTables,freq)
stop

nz,ny,nx=qr.shape
f=89.
umu=np.cos(53/180*3.14)
npol=1
emiss_2d=np.zeros((2,ny,nx),float)
import time

from numba import jit
w_10*=0.75
def fast_Loops(f,umu,emiss_2d,tsk,w_10,T,qv,rho,press,nx,ny,nz,cAlg):
    for i in range(nx):
        for j in range(ny):
            npol=1
            emis=0.8
            #emis,ebar = cAlg.emit(f,npol,tsk[j,i],w_10[j,i],umu)
            emiss_2d[1,j,i]=emis
            npol=0
            #emis,ebar = cAlg.emit(f,npol,tsk[j,i],w_10[j,i],umu)
            emiss_2d[0,j,i]=emis
            ireturn=0
            #for k in range(nz):
            #    abs_air,abs_wv = cAlg.gasabsr98(f,T[k,j,i],qv[k,j,i]*rho[k,j,i],press[k,j,i],ireturn)

fast_Loops(f,umu,emiss_2d,tsk,w_10,T,qv,rho,press,nx,ny,nz,cAlg)
t1=time.time()
abs_air,abs_wv,abs_clw = cAlg.absorption3d(T,qv,qc,rho,press,f)
print(time.time()-t1)
kext=abs_air+abs_wv+0.75*abs_clw
salb=kext.copy()*0.0
asym=kext.copy()*0.0
tb2d = cAlg.radtran3d(T,tsk,kext,salb,asym,emiss_2d,z,umu)

#stop
#absair,abswv = sdsu.gasabsr98(freq,temp,rhowv*1e-3,pa*1e2)
#call gcloud(freq35,tavg,cc(i,j),cld_extKa(i,j))
 
import cartopy
import cartopy.crs as ccrs
fig=plt.figure()
proj=ccrs.PlateCarree()
ax=fig.add_axes([0.1,0.1,0.8,0.8],projection=proj)
plt.pcolormesh(lon,lat,tb2d[0,:,:],cmap='jet')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#plt.xlim(75,85)
#plt.ylim(-60,-50)
plt.colorbar(orientation='horizontal')
#dbz=wrf.getvar(ncFile,'dbz',-1)
#dbzm=np.ma.array(dbz,mask=dbz<-10)
#plt.contour(lon,lat,dbzm[0,:,:],levels=[10,20,30])
#plt.contour(lon,lat,tsk,levels=[273],color='black')

