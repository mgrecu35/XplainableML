
from numpy import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
import combAlg as cmb
cmb.mainfortpy()
cmb.initp2()

from randField3D import *

from netCDF4 import Dataset
fh=Dataset('zKuPrecip_NUBF_Dataset.nc')
pRate=fh['pRate'][:]

pRate1=pRate[:,0:64]
pRate1s=pRate1.std(axis=0)
pRate1m=pRate1.mean(axis=0)

#stop
zHailCoeff=np.array([1.55411351, 2.67654254])
zRainCoeff=np.array([1.49046514, 2.30539888])
zHailCoeff=np.array([1.01036816, 2.4621141])
zHailCoeff=np.array([1.055,3.187])
zRainCoeff=np.array([1.38135457, 2.15647353])
import xarray as xr
sizeX=256

hmCoeff=np.array([ 0.89073539, -1.42430199])
rmCoeff=np.array([ 0.80854362, -1.13862515])
iexist=1


x=np.arange(550)


def getDm():
    field2D_2=gaussian_random_field(beta=-2.35,size = 128)
    field2D_2=field2D_2.real
    ind=np.argmax(field2D_2)
    i0=int(ind/128)
    j0=ind-i0*128
    di=i0-64
    field2D_2=np.roll(field2D_2,64-i0,axis=0)
    field2D_2=np.roll(field2D_2,64-j0,axis=1)
    Dm=0.8+20*field2D_2
    Dm[Dm<0.2]=0.2
    ind=((1.5*Dm-0.5)/0.04).astype(int)
    ind[ind<0]=0
    ind[ind>50]=50
    return 1.5*Dm, ind, 0.35*nwdm[ind,1]+0.65*7.2

def getlwc():
    field2D_2=gaussian_random_field(beta=-2.95,size = 128)
    field2D_2=field2D_2.real
    ind=np.argmax(field2D_2)
    i0=int(ind/128)
    j0=ind-i0*128
    field2D_2=np.roll(field2D_2,64-i0,axis=0)
    field2D_2=np.roll(field2D_2,64-j0,axis=1)
    Dm=20*field2D_2
    
    return 1.5*Dm

def getlwc_u():
    field2D_2=gaussian_random_field(beta=-2.35,size = 128)
    field2D_2=field2D_2.real
    ind=np.argmax(field2D_2)
    i0=int(ind/128)
    j0=ind-i0*128
    Dm=0.8+20*field2D_2
    Dm[Dm<0.2]=0.2
    ind=((1.5*Dm-0.5)/0.04).astype(int)
    return 1.5*Dm

#Dm1,ind1,nw1=getDm()
lwc0=getlwc()

lwc0/=lwc0.std()
ysL=[]
pS=np.interp(np.arange(64),[0,30,45,54,64],[0,20,10,2,0])
y2L=[]
fm=np.interp(range(64),[0,45,50,64],[0,0,1,1])
wl=300/13.8
mu=0
from forwardModel import *
zTL=[]
for j in range(35):
    yL=[]
    y=0
    for i in range(1000):
        rng=np.random.random()
        if rng>0.5:
            y+=1
        else:
            y-=1
        #y=0.9*y+0.1*np.random.randn()
        yL.append(y)
    print(np.std(yL))
    p1=np.array(yL[::10])[0:64]/2+pS/2+pRate1m[::-1]*1.0
    #plt.subplot(211)
    #plt.plot(p1*(1-fm))
    graup=p1*(1-fm)
    graup[graup<0.01]=0.01
    mgraup=10**(hmCoeff[0]*np.log10(graup)+hmCoeff[1])
    mu=0
    nwg,zG,att_g,prate_g,\
        kext_g,kscat_g,g_g=calcZkuH(mgraup,Deq,bscat,ext,scat,g,vfall,mu,wl)
    
    #zG=zHailCoeff[0]*np.log10(graup+1e-9)+zHailCoeff[1]
    rain=p1*(fm)
    rain[rain<0.01]=0.01
    mrain=10**(rmCoeff[0]*np.log10(rain)+rmCoeff[1])
    mu=2
    nwr,zR,att_r,prate_r,\
        kext_out_r,kscat_out_r,\
        g_out_r=calcZkuR(mrain,Deq_r,bscat_r,ext_r,scat_r,g_r,\
                         vfall_r,mu,wl,nw_dm)
    
    #zR=zRainCoeff[0]*np.log10(rain+1e-9)+zRainCoeff[1]
    zT=np.log10(10**(0.1*zR)+10**(0.1*zG))
    plt.scatter(zT*10,range(64)[::-1])
    zTL.append(zT*10)
    #stop
    #plt.subplot(212)
    #plt.plot(p1*(fm))
    y2L.append(p1*(1-fm))
    ysL.append(np.std(yL))

plt.xlim(10,55)
plt.figure()
y2L=np.array(y2L)
y2L[y2L<0]=0
plt.plot(np.array(y2L).mean(axis=0))


plt.figure()
zTL=np.array(zTL)
plt.plot(np.array(zTL).mean(axis=0),range(64)[::-1])
plt.xlim(10,55)
stop
pRate1L=[]
lwc0[i,0:64]*=lwc0[i,0:64]*pRate1s/1.
pRate1L.append(pRate1m+lwc0[i,0:64])

pRateS=np.ma.array(np.array(pRate1L),mask=np.array(pRate1L)<0.1)
plt.pcolormesh(pRateS.T,cmap='jet',vmin=0.1,vmax=120)#,norm=matplotlib.colors.LogNorm())
plt.colorbar()
stop

#plt.figure()
#plt.subplot(121)
#plt.pcolormesh(lwc0,cmap='jet')

#for i in range(9):
#    lwci=getlwc_u()
#    lwc0=0.85*lwc0+0.15*lwci

#plt.subplot(122)
#plt.pcolormesh(lwc0,cmap='jet')


#lwc1=np.pi*1e6*10**nw1*(Dm1*1e-3)**4/4**4
#plt.figure()
#plt.pcolormesh(Dm1,cmap='jet')
#plt.colorbar()
#dmCoeff=array([0.25000273, 0.58028139])

#Dm2,ind2,nw2=getDm()
#plt.figure()
#plt.pcolormesh(exp(dmCoeff[1])*Dm1**(dmCoeff[0]),cmap='jet')
#plt.colorbar()

#plt.figure()
#plt.pcolormesh((Dm2+Dm1)*0.5,cmap='jet')
#plt.colorbar()

#lwc1=(Dm1/1.4)**(1/0.15)/15
#plt.figure()
#plt.pcolormesh(lwc1,cmap='jet',norm=matplotlib.colors.LogNorm())
#plt.colorbar()

#plt.figure()
#plt.pcolormesh(lwc1,cmap='jet',vmax=3)
#plt.colorbar()
#plt.figure()
#plt.pcolormesh(0.5*(field2D_2+field2D_21),cmap='jet')


#Nw=rwc.copy()*0+0.08e8
#dm=(4**4*rwc/(np.pi*1e6*Nw))**0.25*1e3 # in mm
