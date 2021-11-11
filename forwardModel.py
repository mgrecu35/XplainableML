from netCDF4 import Dataset
import numpy as np
from numba import jit
from scipy.special import gamma as gam
def read_wrf(fname,it):
    f=Dataset(fname)
    qv=f['qv'][it,:,:,:]    # water vapor
    qr=f['qr'][it,:,:,:]     # rain mixing ratio
    qs=f['qs'][it,:,:,:]     # snow mixing ratio
    qqc=f['qc'][it,:,:,:]    # cloud mixing ratio
    qg=f['qg'][it,:,:,:]   # graupel mixing ratio
    ncr=f['ncr'][it,:,:,:]     # rain mixing ratio
    ncs=f['ncs'][it,:,:,:]     # snow mixing ratio
    ncg=f['ncg'][it,:,:,:]   # graupel mixing ratio
    #z=f['z_coords'][:]/1000.             # height (km)
    th=f['th'][it,:,:,:]   # potential temperature (K)
    prs=f['prs'][it,:,:,:]
    T=th*(prs/100000)**0.286  # Temperature
    t2c=T-273.15
    #stop
    z=np.arange(81)*0.250
    
    R=287.058  #J*kg-1*K-1
    rho=prs/(R*T)
    return qr,qs,qg,ncr,ncs,ncg,rho,z,t2c,f,prs,qv


nw_dm=np.loadtxt("NwDm.txt")

fnameIce='/home/grecu/scatter-1.1/ice-self-similar-aggregates_13-GHz_scat.nc'
fnameRain='/home/grecu/scatter-1.1/liquid-water_13-GHz_scat.nc'

fnameIce35='/home/grecu/scatter-1.1/ice-self-similar-aggregates_35-GHz_scat.nc'
fnameRain35='/home/grecu/scatter-1.1/liquid-water_35-GHz_scat.nc'

def readScatProf(fname):
    fh=Dataset(fname,'r')
    temp=fh['temperature'][:]
    mass=fh['mass'][:]
    fraction=fh['fraction'][:]
    bscat=fh['bscat'][:]*4*np.pi
    Deq=10*(mass*1e3*6/np
            .pi)**(0.333) # in mm
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

temp,mass,fraction,bscat,Deq,ext,scat,g,vfall=readScatProf(fnameIce)
temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r,fh=readScatProfR(fnameRain)
wl=fh['wavelength'][:]*1000
fact=1e6/np.pi**5/0.93*wl**4
import matplotlib.pyplot as plt

tempKa,massKa,fractionKa,bscatKa,DeqKa,extKa,scatKa,gKa,vfallKa=readScatProf(fnameIce35)
tempKa_r,massKa_r,bscatKa_r,DeqKa_r,extKa_r,scatKa_r,gKa_r,vfallKa_r,fh=readScatProfR(fnameRain35)

freq=13.8
freqKa=35.5

wl=300/freq
wlKa=300/freqKa
mu=2
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

                
def calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    print(wl)
    att_total=rwc.copy()*0
    z_total=rwc.copy()*0.0

    a=np.nonzero(rwc>0.01)
    ncr[ncr<0.001]=0.001
    #ncr[a]=10
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nwr=rwc.copy()*0.0
    w_r=rwc[a].copy()*0.0
    z_r=rwc[a].copy()*0.0
    att_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    print(len(a[0]))
    get_Z(rwc[a],nw_r,lambd_r,w_r,z_r,att_r,dm_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
    nwr[a]=np.log10(nw_r)
    
    z_total[a]+=10.**(0.1*z_r)
    att_total[a]+=att_r

    a=np.nonzero(swc>0.01)
    ncs[ncs<0.001]=0.001
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
    get_Z(gwc[a],nw_g,lambd_g,w_g,z_g,att_g,dm_g,Deq[14,:],bscat[-1,14,:],ext[-1,14,:],\
          vfall[14,:],mu,wl)
    nwg[a]=np.log10(nw_g)
        
    z_total[a]+=10.**(0.1*z_g)
    att_total[a]+=att_g
    
    z_total=10*np.log10(z_total+1e-9)
    z_m=np.ma.array(z_total,mask=z_total<-10)
    z_att_m=z_m.copy()
    gett_atten(z_att_m,z_m,att_total,z)
    return z_m,z_att_m, att_total, nwr,nws,nwg, w_r, nw_r, z_r

def calcZkuR(rwc,Deq_r,bscat_r,ext_r,scat_r,g_r,vfall_r,mu,wl,nw_dm):
    Nw=rwc.copy()*0+0.08e8
    dm=(4**4*rwc/(np.pi*1e6*Nw))**0.25*1e3 # in mm
    for i in range(dm.shape[0]):
        inw=int((dm[i]-0.5)/0.04)
        if inw<0:
            inw=0
        if inw>59:
            inw=59
        dnw=0.5*(nw_dm[inw,1]-6.9)+0.0125*np.random.randn()
        Nw[i]=10**(dnw+6.9)
    dm=(4**4*rwc/(np.pi*1e6*Nw))**0.25*1e3 # in mm
    nwr=rwc.copy()*0.0+Nw
    w_r=rwc.copy()*0.0
    prate_r=rwc.copy()*0.0
    z_r=rwc.copy()*0.0
    att_r=rwc.copy()*0.0
    dm_r=rwc.copy()*0.0
    kext_out_r=rwc.copy()*0.0
    kscat_out_r=rwc.copy()*0.0
    g_out_r=rwc.copy()*0.0
    lambd=(4+mu)/dm
    get_Zn(rwc,nwr,lambd,w_r,z_r,att_r,dm,dm_r,\
           prate_r,kext_out_r,kscat_out_r,g_out_r,Deq_r,bscat_r[9,:],ext_r[9,:],\
           scat_r[9,:],g_r[9,:],\
           vfall_r,mu,wl)
    print(rwc.mean())
    print(w_r.mean())
    return nwr,z_r,att_r,prate_r,kext_out_r,kscat_out_r,g_out_r

def calcZkuS(rwc,T,Deq,bscat,ext,vfall,mu,wl):
    Nw=rwc.copy()*0+0.08e8*8/2.
    a=np.nonzero(T>-10)
    b=np.nonzero(T[a]<0)
    #Nw[a][b]=0.08e8*(8-7*(T[a][b]+10)/10.)
    a=np.nonzero(T>0)
    #Nw[a]=0.08e8
    dm=(4**4*rwc/(np.pi*1e6*Nw))**0.25*1e3 # in mm
    nwr=rwc.copy()*0.0+Nw
    w_r=rwc.copy()*0.0
    prate_s=rwc.copy()*0.0
    z_r=rwc.copy()*0.0
    att_r=rwc.copy()*0.0
    dm_r=rwc.copy()*0.0
    w_s=rwc.copy()*0.0
    z_s=rwc.copy()*0.0
    att_s=rwc.copy()*0.0
    dm_s=rwc.copy()*0.0
    lambd=(4+mu)/dm
    get_Zn(rwc,nwr,lambd,w_s,z_s,att_s,dm,dm_s,\
           prate_s,Deq[14,:],bscat[3,14,:],ext[3,14,:],vfall[3,:],mu,wl)
    
    return nwr,z_s,att_s

def nw_lambd(swc,nc,mu):
    rhow=1e6
    lambd=(nc*rhow*np.pi*gam(4+mu)/gam(1+mu)/6.0/swc)**(0.333)  # m-1
    n0=nc*lambd/gam(1+mu) # m-4
    n0*=1e-3 # mm-1 m-3
    lambd*=1e-2 # cm-1
    return n0,lambd

def intDSD(nw,lambd):
    d=np.array(300)*0.1+0.05
    #lwc=np.sum(nw*np.exp(-lambd*0.1*d)*d**3*0.1)
    return lwc

def nw_lambdh(swc):
    #lambd=650*(1e-3*swc)**(-0.19)*1e-2 #cm-1
    #nw=78*20*36570*(1e-3*swc)**(0.24)*1e-3 #mm-1 m-3
    lambd=np.log(88/swc)/3.45
    nw=115*lambd**3.63
    lambd*=10
    print(lambd)
    print(nw)
    for nw1,l1 in zip(nw,lambd):
        lwc=intDSD(nw1,l1)
        print('lwc=',lwc,nw1,l1)
    return nw,lambd
    
def calcZkuH(rwc,Deq,bscat,ext,scat,g,vfall,mu,wl):
    w_r=rwc.copy()*0.0
    z_r=rwc.copy()*0.0
    att_r=rwc.copy()*0.0
    dm_r=rwc.copy()*0.0
    w_s=rwc.copy()*0.0
    prate_s=rwc.copy()*0.0
    z_s=rwc.copy()*0.0
    att_s=rwc.copy()*0.0
    dm_s=rwc.copy()*0.0
    kext_s=rwc.copy()*0.0
    kscat_s=rwc.copy()*0.0
    g_s=rwc.copy()*0.0
    dm=dm_r.copy()
    #nwr,lambd=nw_lambdh(rwc)
    mu=0
    nw=18
    rhoH=0.9
    lambd=10*(nw*np.pi*rhoH*1e-3/rwc)**(0.25)
    #lwc=np.sum(nw*np.exp(-lambd*d)*(1e-3*d)**3*dD)*1e6*np.pi/6*rhoH
    #print(rwc)
    nwr=rwc.copy()*0+nw
    get_ZH(rwc,nwr,lambd,w_s,z_s,att_s,dm_s,prate_s,\
           kext_s,kscat_s,g_s,Deq[18,:],bscat[-1,18,:],ext[-1,18,:],\
           scat[-1,18,:],g[-1,18,:],vfall[14,:],mu,wl,rhoH)
    #print(dm_s)
    return nwr,z_s,att_s,prate_s,kext_s,kscat_s,g_s

def calcZkuG_2m(rwc,ncr,T,Deq,bscat,ext,scat,g,vfall,mu,wl):
    w_r=rwc.copy()*0.0
    z_r=rwc.copy()*0.0
    att_r=rwc.copy()*0.0
    dm_r=rwc.copy()*0.0
    w_s=rwc.copy()*0.0
    prate_s=rwc.copy()*0.0
    z_s=rwc.copy()*0.0
    att_s=rwc.copy()*0.0
    dm_s=rwc.copy()*0.0
    kext_s=rwc.copy()*0.0
    kscat_s=rwc.copy()*0.0
    g_s=rwc.copy()*0.0
    dm=dm_r.copy()
    nwr,lambd=nw_lambd(rwc,ncr,mu)
    get_Z(rwc,nwr,lambd,w_s,z_s,att_s,dm_s,prate_s,\
          kext_s,kscat_s,g_s,Deq[18,:],bscat[-1,18,:],ext[-1,18,:],\
          scat[-1,18,:],g[-1,18,:],vfall[14,:],mu,wl)
    return nwr,z_s,att_s,prate_s,kext_s,kscat_s,g_s

def calcZkuS_2m(rwc,ncr,T,Deq,bscat,ext,scat,g,vfall,mu,wl):
    w_r=rwc.copy()*0.0
    z_r=rwc.copy()*0.0
    att_r=rwc.copy()*0.0
    dm_r=rwc.copy()*0.0
    prate_s=rwc.copy()*0.0
    w_s=rwc.copy()*0.0
    z_s=rwc.copy()*0.0
    att_s=rwc.copy()*0.0
    dm_s=rwc.copy()*0.0
    dm=dm_r.copy()
    kext_s=rwc.copy()*0.0
    kscat_s=rwc.copy()*0.0
    g_s=rwc.copy()*0.0
    nwr,lambd=nw_lambd(rwc,ncr,mu)
    get_Z(rwc,nwr,lambd,w_s,z_s,att_s,dm_s,prate_s,\
          kext_s,kscat_s,g_s,Deq[6,:],bscat[-1,6,:],ext[-1,6,:],\
          scat[-1,6,:],g[-1,6,:],vfall[6,:],mu,wl)
    return nwr,z_s,att_s,prate_s,kext_s,kscat_s,g_s


@jit(nopython=False)
def get_Zn(w,nw,lambd,W,Z,att,dm,dm_out,rrate_out,\
           kext,kscat,g,Deq,bscat,ext,scat,asym,vfall,mu,wl):
    dD=0.05
    rhow=1 #gcm-3
    Dint=np.arange(160)*dD+dD/2.0
    bscatInt=np.interp(Dint,Deq,bscat)
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    scatInt=np.exp(np.interp(Dint,Deq,np.log(scat)))  #m^2
    gInt=np.interp(Dint,Deq,(asym))  #m^2
    fact=1e3/np.pi**5/0.93*wl**4
    nP=W.shape[0]
    f_mu=6/4**4*(4+mu)**(mu+4)/gam(mu+4)
    #print(vfallInt)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        zray=0.0
        rrate=0.0
        for i in range(160):
            d=dD*i+dD/2
            Nd=f_mu*np.exp(-lambd[j]*d)*(d/dm[j])**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm_out[j]=dm_out[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(d) #(g/m3)
            Z[j]=Z[j]+nw[j]*Nd*bscatInt[i]
            vdop=vdop+nw[j]*Nd*bscatInt[i]*vfallInt[i]
            rrate=rrate+nw[j]*Nd*(0.1*d)**3*np.pi/6*vfallInt[i]
            att[j]=att[j]+nw[j]*Nd*extInt[i]*4.343 #(/km)
            kext[j]=kext[j]+nw[j]*Nd*extInt[i] #(/km)
            kscat[j]=kscat[j]+nw[j]*Nd*scatInt[i] #(/km)
            g[j]=g[j]+nw[j]*Nd*scatInt[i]*gInt[i] #(/km)
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        Z[j]=np.log10(Z[j]*fact)*10
        dm_out[j]=dm_out[j]/(W[j]+1e-9)
        rrate_out[j]=rrate*3.6e-3

@jit(nopython=False)
def get_Z(w,nw,lambd,W,Z,att,dm,prate,kext,kscat,g,Deq,bscat,ext,scat,asym,vfall,mu,wl):
    dD=0.05
    rhow=1 #gcm-3
    Dint=np.arange(320)*dD+dD/2.0
    bscatInt=np.interp(Dint,Deq,bscat)
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    scatInt=np.exp(np.interp(Dint,Deq,np.log(scat)))  #m^2
    gInt=np.interp(Dint,Deq,(asym))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    fact=1e6/np.pi**5/0.93*wl**4
    print(W.shape)
    nP=W.shape[0]
    print(nP,mu,fact)
    print(fact)
    #print(bscatInt)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        zray=0.0
        rrate=0.0
        for i in range(120):
            d=dD*i+dD/2
            Nd=np.exp(-lambd[j]*d*0.1)*(0.1*lambd[j]*d)**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm[j]=dm[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(0.1*d) #(g/m3)
            Z[j]=Z[j]+nw[j]*Nd*bscatInt[i]
            vdop=vdop+nw[j]*Nd*bscatInt[i]*vfallInt[i]
            rrate=rrate+nw[j]*Nd*(0.1*d)**3*np.pi/6*vfallInt[i]
            att[j]=att[j]+nw[j]*Nd*extInt[i]*1e3*4.343 #(/km)
            kext[j]=kext[j]+nw[j]*Nd*extInt[i]*1e3 #(/km)
            kscat[j]=kscat[j]+nw[j]*Nd*scatInt[i]*1e3 #(/km)
            g[j]=g[j]+nw[j]*Nd*scatInt[i]*gInt[i]*1e3 #(/km)
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        Z[j]=np.log10(Z[j]*fact)*10
        dm[j]=dm[j]/(W[j]+1e-9)
        nw_d=4.0**4/np.pi/1e1*W[j]/dm[j]**4
        nw[j]=nw_d
        #print(nw_d/nw[j])
        prate[j]=rrate*3.6
    print(W)


#@jit(nopython=False)
def get_ZH(w,nw,lambd,W,Z,att,dm,prate,kext,kscat,g,Deq,bscat,ext,scat,asym,vfall,mu,wl,rhoH):
    dD=0.05
    rhow=rhoH #gcm-3
    Dint=np.arange(320)*dD+dD/2.0
    bscatInt=np.interp(Dint,Deq,bscat)
    extInt=np.exp(np.interp(Dint,Deq,np.log(ext)))  #m^2
    scatInt=np.exp(np.interp(Dint,Deq,np.log(scat)))  #m^2
    gInt=np.interp(Dint,Deq,(asym))  #m^2
    vfallInt=np.interp(Dint,Deq,vfall)
    fact=1e6/np.pi**5/0.93*wl**4
    print(W.shape)
    nP=W.shape[0]
    print(nP,mu,fact)
    print(fact)
    #print(bscatInt)
    for j in range(nP):
        vdop=0
        nc0=0
        Vol=0
        zray=0.0
        rrate=0.0
        for i in range(320):
            d=dD*i+dD/2
            Nd=np.exp(-lambd[j]*d*0.1)*(0.1*lambd[j]*d)**mu*dD #(mm)
            W[j]=W[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow #(g/m3)
            dm[j]=dm[j]+nw[j]*Nd*(0.1*d)**3*np.pi/6*rhow*(0.1*d) #(g/m3)
            Z[j]=Z[j]+nw[j]*Nd*bscatInt[i]
            vdop=vdop+nw[j]*Nd*bscatInt[i]*vfallInt[i]
            rrate=rrate+nw[j]*Nd*(0.1*d)**3*np.pi/6*vfallInt[i]
            att[j]=att[j]+nw[j]*Nd*extInt[i]*1e3*4.343 #(/km)
            kext[j]=kext[j]+nw[j]*Nd*extInt[i]*1e3 #(/km)
            kscat[j]=kscat[j]+nw[j]*Nd*scatInt[i]*1e3 #(/km)
            g[j]=g[j]+nw[j]*Nd*scatInt[i]*gInt[i]*1e3 #(/km)
            nc0=nc0+nw[j]*Nd
            Vol=Vol+nw[j]*Nd*(1e-3*d)**3*np.pi/6
        Z[j]=np.log10(Z[j]*fact)*10
        dm[j]=dm[j]/(W[j]+1e-9)
        nw_d=4.0**4/np.pi/1e1*W[j]/dm[j]**4
        nw[j]=nw_d
        #print(nw_d/nw[j])
        prate[j]=rrate*3.6
    #print(W)
