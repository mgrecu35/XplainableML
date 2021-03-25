import combAlg as cmb
#cmb.mainfortpy()
#cmb.initp2()
from netCDF4 import Dataset

fname='wrfout_d04_2018-08-03_15:00:00.nc'

fname='../LowLevel/wrfout_d03_2018-06-25_03:36:00'
fname='../extract_wrfout_d03_2011-05-20_23_00_00'

def read_wrf(fname,it):
    f=Dataset(fname)
    qv=f['QVAPOR'][it,:,:,:]    # water vapor
    qr=f['QRAIN'][it,:,:,:]     # rain mixing ratio
    qs=f['QSNOW'][it,:,:,:]     # snow mixing ratio
    qc=f['QCLOUD'][it,:,:,:]    # cloud mixing ratio
    qg=f['QGRAUP'][it,:,:,:]   # graupel mixing ratio
    ncr=f['QNRAIN'][it,:,:,:]     # rain mixing ratio
    ncs=f['QNSNOW'][it,:,:,:]     # snow mixing ratio
    ncg=f['QNGRAUPEL'][it,:,:,:]   # graupel mixing ratio
    #z=f['z_coords'][:]/1000.             # height (km)
    th=f['T'][it,:,:,:]+300    # potential temperature (K)
    prs=f['P'][it,:,:,:]+f['PB'][it,:,:,:]  # pressure (Pa)
    T=th*(prs/100000)**0.286  # Temperature
    t2c=T-273.15
    #stop
    z=(f['PHB'][it,:,:,:]+f['PH'][it,:,:,:])/9.81/1000.
    xlat=f['XLAT'][0,:,:]
    xlong=f['XLONG'][0,:,:]
    R=287.058  #J*kg-1*K-1
    rho=prs/(R*T)
    return qr,qs,qg,ncr,ncs,ncg,rho,z
it=0
qr,qs,qg,ncr,ncs,ncg,rho,z=read_wrf(fname,it)


swc=rho*(qs)*1e3
gwc=rho*qg*1e3
rwc=rho*qr*1e3
ncr=ncr*rho*1
ncg=ncg*rho*1
ncs=(ncs)*rho*1

from scipy.special import gamma as gam
import numpy as np

def nw_lambd(swc,nc,mu):
    rhow=1e6
    lambd=(nc*rhow*np.pi*gam(4+mu)/gam(1+mu)/6.0/swc)**(0.333)  # m-1
    n0=nc*lambd/gam(1+mu) # m-4
    n0*=1e-3 # mm-1 m-3
    lambd*=1e-2 # cm-1
    return n0,lambd

from numba import jit

@jit(nopython=True)
def get_Z(nw,lambd,W,Z,att,dm,Deq,bscat,ext,vfall,mu,wl):
    dD=0.05
    rhow=1 #gcm-3
    Dint=np.arange(160)*dD+dD/2.0
    bscatInt=np.interp(Dint,Deq,bscat)
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    fact=1e6/np.pi**5/0.93*wl**4
    print(W.shape)
    nP=W.shape[0]
    print(nP,mu,fact)
    print(fact)
    print(bscatInt)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        for i in range(160):
            d=dD*i+dD/2
            Nd=np.exp(-lambd[j]*d*0.1)*(0.1*lambd[j]*d)**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm[j]=dm[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(0.1*d) #(g/m3)
            Z[j]=Z[j]+nw[j]*Nd*bscatInt[i]
            vdop=vdop+nw[j]*Nd*bscatInt[i]*vfallInt[i]
            att[j]=att[j]+nw[j]*Nd*extInt[i]*1e3 #(/km)1
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        Z[j]=np.log10(Z[j]*fact)*10
        dm[j]=dm[j]/W[j]

fnameIce='../scatter-1.1/ice-self-similar-aggregates_13-GHz_scat.nc'
fnameRain='../scatter-1.1/liquid-water_13-GHz_scat.nc'

def readScatProf(fname):
    fh=Dataset(fname,'r')
    temp=fh['temperature'][:]
    mass=fh['mass'][:]
    fraction=fh['fraction'][:]
    bscat=fh['bscat'][:]*4*np.pi
    Deq=10*(mass*1e3*6/np.pi)**(0.333) # in mm
    ext=fh['ext'][:]
    scat=fh['scat'][:]
    g=fh['g'][:]
    vfall=fh['fall_speed'][:]
    return temp,mass,fraction,bscat,Deq,ext,scat,g,vfall

def readScatProfR(fname):
    fh=Dataset(fname,'r')
    temp=fh['temperature'][:]
    mass=fh['mass'][:]
    bscat=fh['bscat'][:]*4*np.pi
    Deq=10*(mass*1e3*6/np.pi)**(0.333) # in mm
    ext=fh['ext'][:]
    vfall=fh['fall_speed'][:]
    scat=fh['scat'][:]
    g=fh['g'][:]
    return temp,mass,bscat,Deq,ext,scat,g,vfall

temp,mass,fraction,bscat,Deq,ext,scat,g,vfall=readScatProf(fnameIce)
temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r=readScatProfR(fnameRain)
freq=13.8
#freq=94.0
wl=300/freq

att_total=rwc.copy()*0
z_total=rwc.copy()*0.0
a=np.nonzero(rwc>0.01)
mu=2.0

nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
w_r=rwc[a].copy()*0.0
z_r=rwc[a].copy()*0.0
att_r=rwc[a].copy()*0.0
dm_r=rwc[a].copy()*0.0
get_Z(nw_r,lambd_r,w_r,z_r,att_r,dm_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
z_total[a]+=10.**(0.1*z_r)
att_total[a]+=att_r
#stop

a=np.nonzero(swc>0.01)
nw_s,lambd_s=nw_lambd(swc[a],ncs[a],mu)
w_s=swc[a].copy()*0.0
z_s=swc[a].copy()*0.0
att_s=swc[a].copy()*0.0
dm_s=rwc[a].copy()*0.0
get_Z(nw_s,lambd_s,w_s,z_s,att_s,dm_s,Deq[12,:],bscat[-1,12,:],ext[-1,12,:],\
      vfall[12,:],mu,wl)
z_total[a]+=10.**(0.1*z_s)
att_total[a]+=att_s

a=np.nonzero(gwc>0.01)
nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
w_g=gwc[a].copy()*0.0
z_g=gwc[a].copy()*0.0
att_g=gwc[a].copy()*0.0
dm_g=gwc[a].copy()*0.0
get_Z(nw_g,lambd_g,w_g,z_g,att_g,dm_g,Deq[14,:],bscat[-1,14,:],ext[-1,14,:],\
      vfall[14,:],mu,wl)
z_total[a]+=10.**(0.1*z_g)
att_total[a]+=att_g

z_total=10*np.log10(z_total+1e-9)
z_m=np.ma.array(z_total,mask=z_total<-10)
z_att_m=z_m.copy()
@jit(nopython=True)
def gett_atten(z_att_m,z_m,att_tot,z):
    a=np.nonzero(z_m[0,:,:]>0)
    nz=z_att_m.shape[0]
    for i, j in zip(a[0],a[1]):
        pia_tot=0
        for k in range(nz-1,-1,-1):
            if z_m[k,i,j]>-10:
                pia_tot+=att_tot[k,i,j]*(z[k+1,i,j]-z[k,i,j])*4.343
                z_att_m[k,i,j]-=pia_tot
                pia_tot+=att_tot[k,i,j]*(z[k+1,i,j]-z[k,i,j])*4.343
            else:
                z_att_m[k,i,j]-=pia_tot
            
import matplotlib.pyplot as plt
#plt.hist(np.log10(nw_s/0.08))

gett_atten(z_att_m,z_m,att_total,z)
nx=z_m.shape[-1]
plt.pcolormesh(np.arange(nx),z[:-1,0,0],z_att_m[:,250,:]-z_m[:,250,:],vmin=0, vmax=10,cmap='jet')
plt.ylim(0,15)
plt.xlim(300,650)
plt.colorbar()

cfad=np.zeros((50,60),float)

@jit(nopython=True)
def makecfad(z_m,z,cfad):
    a=np.nonzero(z_m>0)
    for i, j, k in zip(a[0],a[1],a[2]):
        i0=int(z_m[i,j,k])
        j0=int((z[i,j,k]+0.1)/0.250)
        if j0<60 and i0<50:
            cfad[i0,j0]+=1
        z1=z[i,j,k]*0.666+z[i+1,j,k]*0.333
        j0=int((z1)/0.250)
        if j0<60 and i0<50:
            cfad[i0,j0]+=1
        z2=z[i,j,k]*0.333+z[i+1,j,k]*0.666
        j0=int((z2)/0.250)
        if j0<60 and i0<50:
            cfad[i0,j0]+=1

makecfad(z_att_m,z,cfad)
#swc=1.0
#rhow=1e6
#n0=0.08e8
#lambd=(n0*rhow*np.pi*gam(4+mu)/6.0/swc)**(0.333)  # m-1
#nc=n0/lambd*gam(1+mu) # m-4
#lambd*=1e-2 # cm-1
plt.figure()
import matplotlib
plt.pcolormesh(cfad.T,norm=matplotlib.colors.LogNorm(),cmap='jet')
