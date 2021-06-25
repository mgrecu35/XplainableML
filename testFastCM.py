import combAlg as cmb
#cmb.mainfortpy()
#cmb.initp2()
from netCDF4 import Dataset

fname='wrfout_d04_2018-08-03_15:00:00.nc'

#fname='../LowLevel/wrfout_d03_2018-06-25_03:36:00'
fname='../cm1r20.2/run/cm1out.nc'
fh=Dataset(fname)

def read_wrf(fname):
    f=Dataset(fname)
    it=-1
    qv=f['qv'][it,:,:,:]    # water vapor
    qr=f['qr'][it,:,:,:]     # rain mixing ratio
    qs=f['qs'][it,:,:,:]     # snow mixing ratio
    qc=f['qc'][it,:,:,:]    # cloud mixing ratio
    qg=f['qg'][it,:,:,:]   # graupel mixing ratio
    ncr=f['ncr'][it,:,:,:]     # rain mixing ratio
    ncs=f['ncs'][it,:,:,:]     # snow mixing ratio
    ncg=f['ncg'][it,:,:,:]   # graupel mixing ratio
    #z=f['z_coords'][:]/1000.             # height (km)
    th=f['th'][it,:,:,:]    # potential temperature (K)
    prs=f['prs'][it,:,:,:]  # pressure (Pa)
    T=th*(prs/100000)**0.286  # Temperature
    t2c=T-273.15
    #stop
    z=f['zh']
    R=287.058  #J*kg-1*K-1
    rho=prs/(R*T)
    return qr,qs,qg,ncr,ncs,ncg,rho,z,T,prs
it=0
qr,qs,qg,ncr,ncs,ncg,rho,z,T,prs=read_wrf(fname)
zf=fh['zf'][:]
swc=rho*(qs)*1e3*1
gwc=rho*qg*1e3*1
rwc=rho*qr*1e3*1.25
ncr=ncr*rho*1
ncg=ncg*rho*0.5
ncs=(ncs)*rho*2
#stop
from scipy.special import gamma as gam
import numpy as np
a=np.nonzero(T>273.15)
ncr[a]=ncr[a]*(1+(T[a]-273.15)*0.125/2)


from numba import jit

from scattering import *

freq=13.8
freqKa=35.5
#freq=94.0
wl=300/freq
wlKa=300/freqKa
mu=0

@jit(nopython=False)
def gett_atten(nz,a0,a1,z_att_m,z_m,att_tot,zf):
    print(a[0])
    for i, j in zip(a0,a1):
        pia_tot=0
        for k in range(nz-1,-1,-1):
            if z_m[k,i,j]>-10:
                pia_tot+=att_tot[k,i,j]*(zf[k+1]-zf[k])*4.343
                z_att_m[k,i,j]-=pia_tot
                pia_tot+=att_tot[k,i,j]*(zf[k+1]-zf[k])*4.343
            else:
                z_att_m[k,i,j]-=pia_tot



def calcZ(rwc,swc,gwc,ncr,ncs,ncg,zf,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    print(wl)
    att_total=rwc.copy()*0
    z_total=rwc.copy()*0.0

    a=np.nonzero(rwc>0.01)
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nwr=rwc.copy()*0.0
    w_r=rwc[a].copy()*0.0
    z_r=rwc[a].copy()*0.0
    att_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    get_Z(rwc[a],nw_r,lambd_r,w_r,z_r,att_r,dm_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
    print(z_r.max())
    print(w_r.max())
    nwr[a]=np.log10(nw_r)
    #print(rwc[a].mean())
    #print(w_r.mean())
    #stop
    z_total[a]+=10.**(0.1*z_r)
    att_total[a]+=att_r

    a=np.nonzero(swc>0.01)
    nw_s,lambd_s=nw_lambd(swc[a],ncs[a],mu)
    nws=rwc.copy()*0.0
    
    w_s=swc[a].copy()*0.0
    z_s=swc[a].copy()*0.0
    att_s=swc[a].copy()*0.0
    dm_s=swc[a].copy()*0.0
    get_Z(swc[a],nw_s,lambd_s,w_s,z_s,att_s,dm_s,Deq[12,:],bscat[-1,12,:],ext[-1,12,:],\
          vfall[12,:],mu,wl)
    nws[a]=np.log10(nw_s)
    z_total[a]+=10.**(0.1*z_s)
    att_total[a]+=att_s
    
    a=np.nonzero(gwc>0.01)
    nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
    nwg=rwc.copy()*0.0
    
    w_g=gwc[a].copy()*0.0
    z_g=gwc[a].copy()*0.0
    att_g=gwc[a].copy()*0.0
    dm_g=gwc[a].copy()*0.0
    get_Z(gwc[a],nw_g,lambd_g,w_g,z_g,att_g,dm_g,Deq[22,:],bscat[-1,22,:],ext[-1,22,:],\
          vfall[22,:],mu,wl)
    nwg[a]=np.log10(nw_g)
        
    z_total[a]+=10.**(0.1*z_g)
    att_total[a]+=att_g
    
    z_total=10*np.log10(z_total+1e-9)
    z_m=np.ma.array(z_total,mask=z_total<-10)
    z_att_m=z_total.data.copy()
    nz=z_att_m.shape[0]
    a=np.nonzero(z_m[0,:,:]>0)
    gett_atten(nz,a[0],a[1],z_att_m,z_total,att_total,zf)
    z_att_m=np.ma.array(z_att_m,mask=z_att_m<-10)
    return z_m,z_att_m, att_total, nwr,nws,nwg, w_r, nw_r, z_r
    
import matplotlib.pyplot as plt
#plt.hist(np.log10(nw_s/0.08))

#z_m,z_att_m=calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,Deq,ext,scat,g,vfall,\
#                  Deq_r,ext_r,scat_r,g_r,vfall_r,wl)

#zka_m,zka_att_m,attKa,nwr,nws,nwg,\
#    w_r, nw_r, z_r=calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
#                         DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa)

[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
                           temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r]=scatTables['13.8']

[tempKa,massKa,fractionKa,bscatKa,DeqKa,extKa,scatKa,gKa,vfallKa,\
                           tempKa_r,massKa_r,bscatKa_r,DeqKa_r,extKa_r,scatKa_r,gKa_r,vfallKa_r]=scatTables['35.5']
z_m,z_att_m,att,nwr,nws,nwg,\
    w_r, nw_r, z_r,\
    =calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,Deq,ext,bscat,scat,g,vfall,\
                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
plt.figure()
plt.pcolormesh(np.arange(600)*.500,np.arange(80)*0.250,z_att_m[:,25,:],cmap='jet',vmin=0,vmax=45)
plt.ylim(0,15)
plt.xlim(0,250)
plt.colorbar()

zka_m,zka_att_m,attKa,nwr,nws,nwg,\
    w_r, nw_r, z_r=calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
                         DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa)



plt.figure()
plt.pcolormesh(np.arange(600)*.500,np.arange(80)*0.250,zka_att_m[:,25,:],cmap='jet',vmin=0,vmax=35)
plt.ylim(0,15)
plt.xlim(0,250)
plt.colorbar()

plt.figure()
plt.pcolormesh(np.arange(600)*.500,np.arange(80)*0.250,(z_att_m-zka_att_m)[:,25,:],cmap='jet',vmin=0,vmax=35)
plt.ylim(0,15)
plt.xlim(0,250)
plt.colorbar()

stop
d={"w":w_r,"nw":nw_r,"z":z_r}
import pickle
pickle.dump(d,open('CM.pklz','wb'))

#stop
a=np.nonzero(z_m[0,:,:]>0)

tData=np.zeros((len(a[0]),60,11),float)
hgrid=0.125+np.arange(60)*0.25

@jit(nopython=True)
def gridData(z_att_m,zka_att_m,att,attKa,rwc,swc,gwc,nwr,nws,nwg,z,T,ai,aj,hgrid,tData):
    ic=0
    n=ai.shape[0]
    for k in range(n):
        i=ai[k]
        j=aj[k]
        zm=(z[:-1,i,j]+z[1:,i,j])*0.5
        zku=np.interp(hgrid,zm,z_att_m[:,i,j])
        zka=np.interp(hgrid,zm,zka_att_m[:,i,j])
        attku=np.interp(hgrid,zm,att[:,i,j])[::-1].cumsum()
        attka=np.interp(hgrid,zm,attKa[:,i,j])[::-1].cumsum()
        rain=np.interp(hgrid,zm,rwc[:,i,j])
        snow=np.interp(hgrid,zm,swc[:,i,j])
        graup=np.interp(hgrid,zm,gwc[:,i,j])
        nrain=np.interp(hgrid,zm,nwr[:,i,j])
        nsnow=np.interp(hgrid,zm,nws[:,i,j])
        ngraup=np.interp(hgrid,zm,nwg[:,i,j])
        temp=np.interp(hgrid,zm,T[:,i,j])
        tData[ic,:,0]=zku
        tData[ic,:,1]=zka
        tData[ic,:,2]=attku[::-1]
        tData[ic,:,3]=attka[::-1]
        tData[ic,:,4]=rain
        tData[ic,:,5]=snow
        tData[ic,:,6]=graup
        tData[ic,:,7]=nrain
        tData[ic,:,8]=nsnow
        tData[ic,:,9]=ngraup
        tData[ic,:,10]=temp
        ic+=1
            
gridData(z_m,z_att_m,att,attKa,rwc,swc,gwc,nwr,nws,nwg,z,T,a[0],a[1],hgrid,tData)

import xarray as xr
a1=np.nonzero(tData[:,0,0]>0)
tData=tData[a1[0],:,:]
tDataX=xr.DataArray(tData)
d=xr.Dataset({"tData":tDataX})
d.to_netcdf("trainingData.nc")
zka=tData[:,:,1].copy()
zka[zka<0]=0
zku=tData[:,:,0].copy()
zku[zku<0]=0
hgrid=np.arange(60)*0.25
plt.plot(zku.mean(axis=0),hgrid)
plt.plot(zka.mean(axis=0),hgrid)
stop
nx=z_m.shape[-1]
plt.pcolormesh(np.arange(nx),z[:-1,0,0],zka_att_m[:,250,:],vmin=0, vmax=35,cmap='jet')
plt.ylim(0,15)
plt.xlim(300,650)
plt.colorbar()

cfad=np.zeros((50,60),float)

@jit(nopython=True)
def makecfad(z_m,z,cfad):
    a=np.nonzero(z_m>0)
    for i, j, k in zip(a[0],a[1],a[2]):
        i0=int(z_m[i,j,k])
        z1=z[i,j,k]*0.5+z[i+1,j,k]*0.5
        j0=int((z1)/0.250)
        if j0<60 and i0<50:
            cfad[i0,j0]+=1

makecfad(zka_att_m,z,cfad)
#swc=1.0
#rhow=1e6
#n0=0.08e8
#lambd=(n0*rhow*np.pi*gam(4+mu)/6.0/swc)**(0.333)  # m-1
#nc=n0/lambd*gam(1+mu) # m-4
#lambd*=1e-2 # cm-1
plt.figure()
import matplotlib
plt.pcolormesh(cfad.T,norm=matplotlib.colors.LogNorm(),cmap='jet')
