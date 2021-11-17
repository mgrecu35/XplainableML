n1=4900
n2=5100
#n1=1700
#n2=1900
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
fname='2A.GPM.DPR.V8-20180723.20180625-S180356-E193630.024566.V06A.HDF5'
fname='../XPlainable/data/2A.GPM.DPR.V8-20180723.20180625-S041042-E054316.024557.V06A.HDF5'
fhC=Dataset('data/2B.GPM.DPRGMI.CORRA2018.20180625-S041042-E054316.024557.V06A.HDF5')
fh=Dataset(fname,'r')
zKu=fh['NS/PRE/zFactorMeasured'][n1:n2,:,]
zKuC=fh['NS/SLV/zFactorCorrected'][n1:n2,:,:]
pRateCMB=fhC['MS/precipTotRate'][n1:n2,:,:]
zKa=fh['MS/PRE/zFactorMeasured'][n1:n2,:,]
bzd=fh['NS/VER/binZeroDeg'][n1:n2,:,]
pType=(fh['NS/CSF/typePrecip'][n1:n2,:,]/1e7).astype(int)
stormTop=(fh['NS/PRE/binStormTop'][n1:n2,:,]/1e7).astype(int)
bcf=fh['NS/PRE/binClutterFreeBottom'][n1:n2,:,]
s0=fh['NS/PRE/sigmaZeroMeasured'][n1:n2,:]
s0Ka=fh['MS/PRE/sigmaZeroMeasured'][n1:n2,:]
pia=fh['NS/SRT/pathAtten'][n1:n2,:]
fhKu=Dataset('data/2A.GPM.Ku.V8-20180723.20180625-S041042-E054316.024557.V06A.HDF5')
srtPIA=fhKu['NS/SRT/pathAtten'][n1:n2,:]
relPIA=fhKu['NS/SRT/reliabFlag'][n1:n2,:]
#stop
relFlag=fh['NS/SRT/reliabFlag'][n1:n2,:]
sfcRain=fh['NS/SLV/precipRateNearSurface'][n1:n2,:]
pRate=fh['NS/SLV/precipRate'][n1:n2,:]
zKum=np.ma.array(zKu,mask=zKu<0)
zKam=np.ma.array(zKa,mask=zKa<0)
plt.subplot(211)
#plt.pcolormesh(zKum[:,35,::-1].T,cmap='jet',vmax=50)
hCoeff=np.array([ 0.06605835, -2.08407732])
hCoeff=np.array([ 0.06737808, -3.34419843])
hCoeff=np.array([ 0.07946967, -2.32328256])

import matplotlib
plt.pcolormesh((10**(hCoeff[0]*zKum[:,35,::-1]+hCoeff[1])).T,\
               cmap='jet',vmax=50,vmin=0.01, norm=matplotlib.colors.LogNorm())
plt.pcolormesh((zKum[:,35,::-1]-zKam[:,23,::-1]).T,\
               cmap='jet',vmax=20,vmin=-5)
plt.xlim(125,175)
plt.ylim(10,100)
#plt.xlim(145,160)
nx,ny,nz=zKum.shape
hailD=np.zeros((nx,ny),int)
hailD2=np.zeros((nx,ny),int)
zKuL=[]
indL=[]
zKaL=[]
bzdL=[]
ihaiL=[]
pRateL=[]
pRateDL=[]
srtPIAL=[]
for i in range(nx):
    for j in range(12,37):
        n1=min(bzd[i,j],bcf[i,j])
        ind=np.nonzero(zKum[i,j,:n1]>40)
        if bcf[i,j]-bzd[i,j]<25:
            continue
        if pType[i,j]==2:
            nzb1=bzd[i,j]
            zKuL.append(zKum[i,j,nzb1-60:nzb1+26])
            zKaL.append(zKam[i,j-12,nzb1-60:nzb1+26])
            nzb=int(bzd[i,j]/2)
            pRateL.append(pRateCMB[i,j-12,nzb-30:nzb+10])
            pRateDL.append(pRate[i,j,nzb1-60:nzb1+26])
            bzdL.append(bzd[i,j])
            srtPIAL.append(srtPIA[i,j])
            indL.append([i,j])
            if len(ind[0])>16:
                #print(ind)
                #stop
                ihaiL.append(1)
                if n1-ind[0][0]>8:
                    hailD[i,j]=1
            else:
                ihaiL.append(0)
                if len(ind[0])<4:
                    hailD2[i,j]=1
                    #else:
            #    if len(ind[0])>4:
            #        hailD2[i,j]=1
        
#stop
import combAlg as cmb
cmb.mainfortpy()
cmb.initp2()

plt.colorbar()
plt.subplot(212)
plt.pcolormesh(zKam[:,23,::-1].T,cmap='jet',vmax=35)
plt.xlim(125,175)
plt.ylim(10,100)
#plt.xlim(145,160)
plt.colorbar()
plt.figure()
pRateL=np.array(pRateL)
a=np.nonzero(hailD2>0)
for i1,j1 in zip(a[0][:],a[1][:]):
    plt.subplot(121)
    plt.plot(zKum[i1,j1,:],range(176))
    plt.ylim(160,60)
    plt.xlim(0,55)
    plt.subplot(122)
    if abs(j1-24)<12:
        plt.plot(zKam[i1,j1-12,:],range(176))
        plt.ylim(160,60)
    #plt.show()
from minisom import MiniSom
zKuL=np.array(zKuL)
zKaL=np.array(zKaL)
n1=10
n2=1
nz=86
zKuL[zKuL<0]=0
zKaL[zKaL<0]=0
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
#np.random.seed(seed=10)
som.random_weights_init(zKuL)
som.train_random(zKuL,500) # training with 100 iterations
nt=zKuL.shape[0]
winL=np.zeros((nt),int)
it=0
for z1 in zKuL:
    win=som.winner(z1)
    winL[it]=win[0]
    it+=1

attKuGCoeff=np.array([ 1.01772423, -5.3065495])
attKuRCoeff=np.array([ 0.72389776, -3.2524177 ])

attKup=10**attKuGCoeff[1]*10**(0.1*cmb.tablep2.zkur[0:270]*attKuGCoeff[0])

rrateKuCoeff=np.polyfit(cmb.tablep2.zkur[:270]/10,np.log10(cmb.tablep2.rainrate[:270]),1)
plt.figure()
grauprateKuCoeff=np.polyfit(cmb.tablep2.zkur[:270]/10,np.log10(cmb.tablep2.grauprate[:270]),1)
#stop
plt.scatter(cmb.tablep2.attkug[:270],attKup)

dfrKug=cmb.tablep2.zkur[:270]-cmb.tablep2.zkag[:270]
dmG=cmb.tablep2.dmg[:270]
graupRate=cmb.tablep2.grauprate[:270]
S=10**(cmb.tablep2.zkur[:270]*0.1*0.45)*0.32
plt.figure()
#plt.plot(graupRate)
plt.plot(np.log10(graupRate/S))



ihaiL=np.array(ihaiL)
from bisect import *
pRateDL=np.array(pRateDL)
dmL=[]
nwL=[]
snowRateL=[]

from forwardModel import *
nw=180
rwc=np.array(np.arange(200)*0.04+0.02)
mu=0
wlKu=300/13.8
wlKa=300/35.5
#nwg,z_g,att_g,prate_g,\
#    kext_g,kscat_g,g_g,dm_g=calcZkuG(nw,rwc,Deq,bscat,ext,scat,g,vfall,mu,wlKu)

#nwka_g,zka_g,attka_g,prateka_g,\
#    kextka_g,kscatka_g,gka_g,dmka_g=calcZkuG(nw,rwc,DeqKa,bscatKa,
#                                      extKa,scatKa,gKa,vfallKa,mu,wlKa)
#stop
a=48.34
a=300
b=0.8
b=1.4
dr=0.125
alpha=10**attKuGCoeff[1]
beta=attKuGCoeff[0]
alphaR=10**attKuRCoeff[1]
betaR=attKuRCoeff[0]
#beta=1.0
def hb(zKum,alpha,beta,dr):
    q=0.2*np.log(10)
    zeta=q*beta*alpha*10**(0.1*zKum*beta)*dr
    srt_piaKu=4.0
    zetamax=1.-10**(-srt_piaKu/10.*beta)
    if zeta.cumsum()[-1]>zetamax:
        eps=0.9999*zetamax/zeta.cumsum()[-1]
        #zeta=eps*zeta
    else:
        eps=1.0
    corrc=eps*zeta.cumsum()
    zc=zKum-10/beta*np.log10(1-corrc)
    return zc,eps,-10/beta*np.log10(1-corrc[-1])

def hbR(zKum,alpha,beta,dr):
    q=0.2*np.log(10)
    zeta=q*beta*alpha*10**(0.1*zKum*beta)*dr
    srt_piaKu=zKum.max()-zKum[-1]+2
    
    zetamax=1.-10**(-srt_piaKu/10.*beta)
    if zeta.cumsum()[-1]>zetamax:
        eps=0.99999*zetamax/zeta.cumsum()[-1]
        zeta=eps*zeta
    else:
        eps=1.0
    #corrc=eps*zeta.cumsum()
    zc=zKum-10/beta*np.log10(1-zeta.cumsum())
    print(srt_piaKu,-10/beta*np.log10(1-zeta.cumsum()[-1]),eps)
    return zc,eps,-10/beta*np.log10(1-zeta.cumsum()[-1])
epsL=[]
x1,x2=[],[]
for i in range(0,6):
    aw=np.nonzero(winL==i)
    plt.figure()
    plt.subplot(131)
    plt.plot(zKuL[aw[0],:101].mean(axis=0),range(nz))
    SL=[]
    piaL=[]
    for iw in aw[0]:
        #print(zKuL[iw,57:59],zKuL[iw,64:67])
        dfr1=zKuL[iw,59]-zKaL[iw,59]
        ibin=bisect(dfrKug,dfr1)
        #S=(10**(0.1*zKuL[iw,:])/a)**(1/b)
        S=(10**(0.1*zKuL[iw,:]*0.45))*0.32
        S=1.7*10**(0.1*zKuL[iw,:]*grauprateKuCoeff[0])*10**grauprateKuCoeff[1]

        #attKu1d=10**attKuCoeff[1]*10**(0.1*zKuL[iw,:]*attKuCoeff[0])
        zcG,eps,piaHB=hb(zKuL[iw,:60],alpha,beta,dr)
        zKuL[iw,60:]+=piaHB
        zcR,eps,piaHBR=hbR(zKuL[iw,60:],alphaR,betaR,dr)
        #print(piaHB,zc[-1]-zKuL[iw,59])
        piaL.append(piaHB+piaHBR)#zc[-1]-zKuL[iw,60])#attKu1d[:60].sum()*0.125*2)
        print(piaHB+piaHBR,srtPIAL[iw])
        x1.append(piaHB+piaHBR)
        x2.append(srtPIAL[iw])
        epsL.append(eps)
        nw1=(zKuL[iw,59]-cmb.tablep2.zkur[ibin])/10.
        if ibin>269:
            ibin=269
        #print(nw1,dmG[ibin],dfr1)
        nwL.append(nw1)
        dmL.append(dmG[ibin])
        epsR=S[60:]*0+eps
        epsR[0:3]=np.array([1.2,1.1,1])*eps
        #snowRateL.append(10**(0.1*nw1)*graupRate[ibin])
        S[60:]=epsR**((1-rrateKuCoeff[0])/(1-betaR))*10**rrateKuCoeff[1]*10**(0.1*zcR*rrateKuCoeff[0])
        #S[:60]=zcG
        #S[60:]=zcR
        SL.append(S)

        #plt.figure()
        #plt.plot(S,range(86))
        #plt.ylim(85,0)
        #plt.show()
        #stop
    print(np.mean(piaL),np.mean(epsL))
    SL=np.array(SL)
    plt.plot(zKuL[aw[0],60:61].mean(axis=0),range(60,61),'*')
    plt.ylim(nz-1,0)
    plt.title(ihaiL[aw[0]].sum()/len(aw[0]))
    plt.subplot(132)
    plt.title('class# %i'%(i+1))
    plt.plot(zKuL[aw[0],:90].mean(axis=0)-zKaL[aw[0],:90].mean(axis=0),range(nz))
    plt.ylim(nz-1,0)
    plt.subplot(133)
    plt.title('# profiles %i'%(len(aw[0])))
   # plt.plot(pRateDL[aw[0],:].mean(axis=0),np.arange(nz))
    plt.plot(np.array(SL).mean(axis=0),range(nz))
        
    plt.ylim(nz,0)
stop
plt.figure()
plt.plot(range(125,175),10-s0[125:175,23])
plt.plot(range(125,175),0-s0Ka[125:175,11])
plt.plot(range(125,175),10-s0[125:175,23])
plt.plot(range(125,175),(s0[125:175,22]-s0Ka[125:175,11]-2)/6.)
plt.xlim(145,160)


plt.figure()
plt.subplot(211)
plt.pcolormesh(zKum[155,:,::-1].T,cmap='jet',vmax=50)
plt.colorbar()
plt.subplot(212)
hCoeff=np.array([ 0.06605835, -2.38407732])
plt.pcolormesh((10**(hCoeff[0]*zKum[160,:,::-1]+hCoeff[1])).T,\
               cmap='jet',vmax=10)
plt.colorbar()
#stop
#stop
pRate=fh['NS/SLV/precipRate'][n1:n2,:]
zKu[zKu<0]=0
zmL2=[]
pRateL=[]
piaL=[]
relFlagL=[]
sfcRainL=[]

hCoeff=np.array([ 0.06605835, -2.38407732])

stop
for i1 in range(zKu.shape[0]):
    for j1 in range(20,28):
        if bzd[i1,j1]>stormTop[i1,j1]+4 and bcf[i1,j1]-bzd[i1,j1]>20 and\
           pType[i1,j1]==2:
            if bzd[i1,j1]-60>0 and bzd[i1,j1]+20<176:
                zmL2.append(zKu[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+20])
                pRateL.append(pRate[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+30])
                piaL.append(pia[i1,j1])
                relFlagL.append(relFlag[i1,j1])
                sfcRainL.append(sfcRain[i1,j1])

import pickle
som=pickle.load(open("miniSOM_Land.pklz","rb"))

nx=len(zmL2)
iclassL=[]
for it in range(nx):
    win=som.winner(zmL2[it])
    iclass=win[0]*3+win[1]+1
    iclassL.append(iclass)

plt.figure()
plt.plot(np.array(pRateL).mean(axis=0),-60+np.arange(90))
plt.ylim(30,-60)
plt.xlim(0,40)

plt.figure()

for i in range(nx):
    if iclassL[i]==9:
        plt.plot(zmL2[i],-60+np.arange(80))
plt.ylim(20,-60)

from sklearn.cluster import KMeans

plt.figure(figsize=(12, 12))

iclassL=np.array(iclassL)
a=np.nonzero(iclassL==9)
# Incorrect number of clusters
zmL2=np.array(zmL2)

kmeans = KMeans(n_clusters=16, random_state=10).fit(zmL2[a[0],:])
plt.figure()
zmAvg=[]
piaL=np.array(piaL)
sfcRainL=np.array(sfcRainL)
for i in range(16):
    a1=np.nonzero(kmeans.labels_==i)
    #plt.figure()
    zm1=zmL2[a[0][a1],:].mean(axis=0)
    if zm1.max()>47:
        plt.plot(zm1,-60+np.arange(80))
    zmAvg.append(zmL2[a[0][a1],:].mean(axis=0))
    plt.ylim(20,-60)
    #plt.title("PIA=%6.2f %6.2f"%(piaL[a[0][a1]].mean(),sfcRainL[a[0][a1]].mean()))
plt.xlabel('dBZ')
plt.ylabel('Relative range')
plt.savefig('deepConvProfs.png')
pickle.dump({"zmAvg":zmAvg},open("zmAvg.pklz","wb"))
