from numba import jit
import numpy as np
from netCDF4 import Dataset
from scipy.special import gamma as gam

def nw_lambd(swc,nc,mu):
    rhow=1e6
    lambd=(nc*rhow*np.pi*gam(4+mu)/gam(1+mu)/6.0/swc)**(0.333)  # m-1
    n0=nc*lambd/gam(1+mu) # m-4
    n0*=1e-3 # mm-1 m-3
    lambd*=1e-2 # cm-1
    return n0,lambd

from numba import jit

@jit(nopython=True)
def get_scatt(W,nw,lambd,nw_out,kext,kscat,g,dm,Deq,ext,scat,asym,vfall,mu,wl):
    dD=0.05
    rhow=1 #gcm-3
    Dint=np.arange(160)*dD+dD/2.0
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    scatInt=np.exp(np.interp(Dint,Deq,np.log(scat)))  #m^2
    gInt=np.interp(Dint,Deq,(asym))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    #print(bscatInt)
    nP=W.shape[0]
    print(W.shape)
    print(nw.shape)
    print(lambd.shape)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        for i in range(160):
            d=dD*i+dD/2
            Nd=np.exp(-lambd[j]*d*0.1)*(0.1*lambd[j]*d)**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm[j]=dm[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(0.1*d) #(g/m3)
            kext[j]=kext[j]+nw[j]*Nd*extInt[i]*1e3 #(/km)
            kscat[j]=kscat[j]+nw[j]*Nd*scatInt[i]*1e3 #(/km)
            g[j]=g[j]+nw[j]*Nd*scatInt[i]*gInt[i]*1e3 #(/km)
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        dm[j]=dm[j]/(W[j]+1e-9)
        nw_d=4.0**4/np.pi/1e1*W[j]/dm[j]**4
        nw_out[j]=nw_d
        g[j]=g[j]/kscat[j]


@jit(nopython=True)
def get_Z(w,nw,lambd,W,Z,att,dm,Deq,bscat,ext,vfall,mu,wl):
    dD=0.05
    rhow=1 #gcm-3
    Dint=np.arange(160)*dD+dD/2.0
    bscatInt=np.interp(Dint,Deq,bscat)
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    fact=1e6/np.pi**5/0.93*wl**4
    nP=W.shape[0]
    #print(nP,mu,fact)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        zray=0
        for i in range(160):
            d=dD*i+dD/2
            Nd=np.exp(-lambd[j]*d*0.1)*(0.1*lambd[j]*d)**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm[j]=dm[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(0.1*d) #(g/m3)
            Z[j]=Z[j]+nw[j]*Nd*bscatInt[i]
            zray+=nw[j]*Nd*(d)**6
            vdop=vdop+nw[j]*Nd*bscatInt[i]*vfallInt[i]
            att[j]=att[j]+nw[j]*Nd*extInt[i]*1e3 #(/km)1
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        Z[j]=np.log10(Z[j]*fact)*10
        dm[j]=dm[j]/(W[j]+1e-9)
        nw_d=4.0**4/np.pi/1e1*W[j]/dm[j]**4
        nw[j]=nw_d
        if(W[j]>10.5):
            print(W[j],np.log10(zray)*10,Z[j],np.log10(zray)*10-Z[j])
        #print(nw[j],nw_d,W[j],w[j])




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
    #print(fh)
    #stop
    return temp,mass,bscat,Deq,ext,scat,g,vfall,fh

scatTables={}
for freq in [13.8,35.5,89,166,183.3]:
    fnameIce='../scatter-1.1/test/scattering_input/ice-self-similar-aggregates_%06.2f-GHz_scat.nc'%freq
    fnameRain='../scatter-1.1/test/scattering_input/liquid-water_%06.2f-GHz_scat.nc'%freq
    temp,mass,fraction,bscat,Deq,ext,scat,g,vfall=readScatProf(fnameIce)
    temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r,fh=readScatProfR(fnameRain)
    scatTables[str(freq)]=[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
                           temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r]

    
def calcScatt(rwc,swc,gwc,ncr,ncs,ncg,scatTables,freq):
    mu=2.0
    temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
        temp_r,mass_r,bscat_r,Deq_r,ext_r,\
        scat_r,g_r,vfall_r=scatTables[str(freq)]
    
    kext_total=rwc.copy()*0
    kscat_total=rwc.copy()*0.0
    nws_out=swc.copy()
    g_total=rwc.copy()*0.0
    a=np.nonzero(rwc>0.01)
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nw_r_out=rwc[a].copy()*0.0
    kext_r=rwc[a].copy()*0.0
    kscat_r=rwc[a].copy()*0.0
    g_fact_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    Wr=rwc[a].copy()*0.0
    wl=300/freq
    get_scatt(Wr,nw_r,lambd_r,nw_r_out,kext_r,kscat_r,g_fact_r,dm_r,Deq_r,\
              ext_r[9,:],scat_r[9,:],g_r[9,:],vfall_r,mu,wl)

    kext_total[a]+=kext_r
    kscat_total[a]+=kscat_r
    g_total[a]+=kscat_r*g_fact_r
    
    a=np.nonzero(swc>0.01)
    nw_s,lambd_s=nw_lambd(swc[a],ncs[a],mu)
    nw_s_out=rwc[a].copy()*0.0
    kext_s=swc[a].copy()*0.0
    kscat_s=swc[a].copy()*0.0
    g_fact_s=swc[a].copy()*0.0
    dm_s=swc[a].copy()*0.0
    Ws=swc[a].copy()*0.0
    wl=300/freq
    
    get_scatt(Ws,nw_s,lambd_s,nw_s_out,kext_s,kscat_s,g_fact_s,dm_s,Deq[11,:],\
              ext[-1,11,:],scat[-1,11,:],g[-1,11,:],vfall[11,:],mu,wl)
    print('ratio=',nw_s.mean()/nw_s_out.mean())
    #stop
    nws_out[a]=nw_s_out
    kext_total[a]+=kext_s
    kscat_total[a]+=kscat_s
    g_total[a]+=kscat_s*g_fact_s
    
    a=np.nonzero(gwc>0.01)
    nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
    nw_g_out=rwc[a].copy()*0.0
    kext_g=gwc[a].copy()*0.0
    kscat_g=gwc[a].copy()*0.0
    g_fact_g=gwc[a].copy()*0.0
    dm_g=gwc[a].copy()*0.0
    Wg=gwc[a].copy()*0.0
    wl=300/freq
    
    get_scatt(Wg,nw_g,lambd_g,nw_g_out,kext_g,kscat_g,g_fact_g,dm_g,Deq[11,:],\
              ext[-1,14,:],scat[-1,14,:],g[-1,14,:],vfall[11,:],mu,wl)

    kext_total[a]+=kext_g
    kscat_total[a]+=kscat_g
    g_total[a]+=kscat_g*g_fact_g
    return kext_total,kscat_total, g_total, nws_out, dm_s
