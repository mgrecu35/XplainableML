import combAlg as cmb
#cmb.mainfortpy()
#cmb.initp2()
from netCDF4 import Dataset
import wrf


#fname='../LowLevel/wrfout_d03_2018-06-25_03:36:00'
fname='../extract_wrfout_d03_2011-05-20_23_00_00'
fname='/media/grecu/ExtraDrive1/1April2018/wrfout_d03_2018-03-31_21:00:00'
fname='wrfout_d04_2018-08-03_15:00:00.nc'

from wrf_io import read_wrf
import combAlg as cAlg

it=-1
qr,qs,qg,ncr,ncs,ncg,qc,rho,z,T,tsk,w_10,qv,press,z=read_wrf(fname,it)
ncFile=Dataset(fname)
dbz=wrf.getvar(ncFile,'dbz',-1)
lon=ncFile.variables['XLONG'][0,:,:]
lat=ncFile.variables['XLAT'][0,:,:]
swc=rho*(qs)*1e3
gwc=rho*qg*1e3
rwc=rho*qr*1e3
ncr=ncr*rho*1
ncg=ncg*rho*1
ncs=(ncs)*rho*1
import numpy as np
import pickle
from sklearn import neighbors
ml=1
if ml==1:
    d=pickle.load(open("simTbs_3789_08_03.pklz","rb"))
    tb_37=d["tb_35.5"]
    tb_89=d["tb_89"]
    a=np.nonzero(dbz[10,:,:].data>10)
    n_neighbors = 50
    X=[]
    y=[]
    Xv=[]
    yv=[]
    
    
    for i in range(len(a[0])):
        x1=[tb_37[0,a[0][i],a[1][i]],tb_37[1,a[0][i],a[1][i]],tb_89[0,a[0][i],a[1][i]],tb_89[0,a[0][i],a[1][i]]]
        y1=rwc[0,a[0][i],a[1][i]]
        r=np.random.random()
        if r>0.5:
            Xv.append(x1)
            yv.append(y1)
        else:
            X.append(x1)
            y.append(y1)
        
    weights='distance'
    X=np.array(X)
    y=np.array(y)
    Xv=np.array(Xv)
    yv=np.array(yv)
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(Xv)
    
    stop

from scipy.special import gamma as gam
import numpy as np
a=np.nonzero(T>273.15)
#ncr[a]=ncr[a]*(1+(T[a]-273.15)*0.25)
ncr[ncr<0.0001]=0.0001
ncs[ncs<0.0001]=0.0001
ncg[ncg<0.0001]=0.0001
print(ncr.min())
from numba import jit

nz,ny,nx=qr.shape
freq=37
umu=np.cos(53/180*3.14)
npol=1
emiss_2d=np.zeros((2,ny,nx),float)
import time
import combAlg as cAlg

from numba import jit
w_10*=0.75

def fast_Loops(f,umu,emiss_2d,tsk,w_10,T,qv,rho,press,nx,ny,nz,cAlg):
    for i in range(nx):
        for j in range(ny):
            npol=1
            emis,ebar = cAlg.emit(f,npol,tsk[j,i],w_10[j,i],umu)
            emiss_2d[1,j,i]=emis
            npol=0
            emis,ebar = cAlg.emit(f,npol,tsk[j,i],w_10[j,i],umu)
            #emis=0.65+0.25*np.random.rand()
            emiss_2d[0,j,i]=emis
            ireturn=0
            #for k in range(nz):
            #    abs_air,abs_wv = cAlg.gasabsr98(f,T[k,j,i],qv[k,j,i]*rho[k,j,i],press[k,j,i],ireturn)

import pickle
#{"emiss":emiss_2d},

fast_Loops(freq,umu,emiss_2d,tsk,w_10,T,qv,rho,press,nx,ny,nz,cAlg)
pickle.dump({"emiss":emiss_2d},open("emiss2d_37.pklz","wb"))
#d=pickle.load(open("emiss2d.pklz","rb"))
#emiss_2d=d["emiss"]
def simTb(freq,freqv,T,tsk,qv,qc,qs,qg,qr,rho,press,emiss_2d):
    t1=time.time()
    abs_air,abs_wv,abs_clw = cAlg.absorption3d(T,qv,qc,rho,press,freqv)
    print(time.time()-t1)
    kext_atm=abs_air+abs_wv
    kext=abs_air+abs_wv+0.75*abs_clw
    salb0=kext.copy()*0.0
    asym0=kext.copy()*0.0
    tb2d_clear = cAlg.radtran3d(T,tsk,kext,salb0,asym0,emiss_2d,z,umu)
    print(freq)
    kext_hyd,kscat_hyd, g_hyd,nws_out, dm_s=\
        calcScatt(rwc,swc,gwc,ncr,ncs,ncg,scatTables,freq)
    kext_tot=kext+kext_hyd
    salb=kscat_hyd/kext_tot
    asym=g_hyd/kext_tot
    tb2d = cAlg.radtran3d(T,tsk,kext_tot,salb,asym,emiss_2d,z,umu)
    return tb2d, tb2d_clear,nws_out, dm_s, kext_atm

freqvL=[35.5,89]
freqL=[35.5,89]
from scattering import *
d={}
import xarray as xr
for freqv,freq in zip(freqvL,freqL):
    tb2d,tb2d_clear,nws_out,dm_s,kext_atm=simTb(freq,freqv,T,tsk,qv,qc,qs,qg,qr,rho,press,emiss_2d)
    d["tb_"+str(freqv)]=xr.DataArray(tb2d.copy())

import cartopy
import cartopy.crs as ccrs
proj=ccrs.PlateCarree()
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8],projection=proj)
c=plt.pcolormesh(lon,lat,2.1*tb2d[1,:,:]-1.1*tb2d[0,:,:],cmap='jet',transform=proj)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
plt.xlim(125,140)
plt.ylim(-54,-44)
plt.colorbar(orientation='horizontal')
import pickle
pickle.dump(d,open("simTbs_3789_08_03.pklz","wb"))
