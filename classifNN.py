from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
K = keras.backend

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

fh=Dataset("NH_v2L.nc")

zKu=fh['zKuLx'][:,:]
zKu[zKu<-10]=-10
XKu=(zKu).copy()
bzd=fh['bzdL'][:]

zKa=fh['zKaLx'][:,:]
zKa[zKa<-10]=-10
XKa=(zKa).copy()
bzd=fh['bzdL'][:]
pRate=fh['pRate_cmb'][:,:]
pRate[pRate<0]=0
pRateDPR=fh['pRate'][:,:]
pRateDPR[pRateDPR<0]=0
XpRate=pRate.copy()
XpRate_DPR=pRateDPR.copy()
from numba import jit
cfadKa=np.zeros((50,68))
cfadKu=np.zeros((50,68))
pType=fh['pType'][:]
top=fh['stormTop'][:]-100
top[top<0]=0
@jit(nopython=True)
def makecfad(cfadKa,zKu,pType,bzd,top):
    nx,nz=zKa.shape
    for i in range(nx):
        if pType[i]==2 and (bzd[i]-136)*(bzd[i]-140)<0:
            for j in range(top[i],68):
                i0=int(zKu[i,j])
                if i0>=0 and i0<50:
                    cfadKa[i0,j]+=1
            for j in range(0,top[i]):
                zKu[i,j]=-10
import matplotlib
makecfad(cfadKu,XKu,pType,bzd,top)
plt.pcolormesh(cfadKu[:,::-1].T, norm=matplotlib.colors.LogNorm(),cmap='jet')

plt.figure()
makecfad(cfadKa,XKa,pType,bzd,top)
plt.pcolormesh(cfadKa[:,::-1].T, norm=matplotlib.colors.LogNorm(),cmap='jet')

def prepareData(nz1,nz2,X1,X2,Y,Y2,bzd,batch_size,pType,top):
    scalerX=StandardScaler()
    scalerX2=StandardScaler()
    scalerY=StandardScaler()
    a=np.nonzero((bzd-136)*(bzd-140)<0)
    b=np.nonzero(pType[a]==1)
    c=np.nonzero(100+top[a][b]<bzd[a][b]-8)
    #X_sub=scalerX.fit_transform(X[a[0][b],:nz1])
    Y_sub=scalerY.fit_transform(Y[a[0][b],:nz2])
    print(len(b[0]))
    print(len(c[0]))
    Y_sub=(Y[a[0][b][c],:nz2])
    Y2_sub=(Y2[a[0][b][c],:nz1])
    X1_sub=(X1[a[0][b][c],:nz1])
    X2_sub=(X2[a[0][b][c],:nz1])
    n=X1_sub.shape[0]
    r=np.random.random(n)
    a=np.nonzero(r>0.00)
    b=np.nonzero(r<0.25)
    print(Y2_sub.shape)
    print(Y_sub.shape)
    X1_train=X1_sub[a[0],:nz1].copy()
    X1_valid=X1_sub[b[0],:nz1].copy()
    X2_train=X2_sub[a[0],:nz1].copy()
    X2_valid=X2_sub[b[0],:nz1].copy()
    Y_train=Y_sub[a[0],:nz2].copy()
    Y_valid=Y_sub[b[0],:nz2].copy()
    Y2_train=Y2_sub[a[0],:nz1].copy()
    Y2_valid=Y2_sub[b[0],:nz1].copy()
    ind=np.arange(X1_train.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices(ind).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    return X1_train,X1_valid, X2_train,X2_valid, Y_train, Y_valid, Y2_train, Y2_valid, dataset



batch_size=128
X_train,X_valid,XKu_train,XKu_valid,\
    Y_train, Y_valid,\
    Y2_train, Y2_valid,\
    datasetS=prepareData(68,34,XKa,XKu,XpRate,XpRate_DPR,bzd,batch_size,pType,top)
nc=16
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=nc, random_state=0).fit(X_train)
#for c in  range(nc):
plt.figure()
#or c in range(16):
#Refined microphyiscal parameterization to ehnance and extend the GPM combined algorithm
for c in range(nc):
    #if c%4==0:
    plt.figure()
    plt.suptitle('Stratiform class#%2i'%c)
    #i=c-int(c/4)*4
    plt.subplot(1,2,1)
    a=np.nonzero(kmeans.labels_==c)
    plt.plot(XKu_train[a[0],:].mean(axis=0)[::-1],7*0.125+np.arange(68)*0.125)
    plt.plot(X_train[a[0],:].mean(axis=0)[::-1],7*0.125+np.arange(68)*0.125)
    plt.xlim(0,45)
    #plt.title('Class %2i, nc=%i'%(c,len(a[0])))
    plt.ylabel('Height (km)')
    plt.legend(['Ku-band','Ka-band'])
    plt.xlabel('dBZ')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(Y_train[a[0],:].mean(axis=0)[::-1][1:],7*0.125+0.25+np.arange(33)*2*0.125)
    plt.plot(Y2_train[a[0],:].mean(axis=0)[::-1][:],7*0.125+np.arange(68)*0.125)
    plt.legend(['CMB','DPR'])
    plt.xlabel('mm/h')
    plt.grid(True)
    plt.savefig('Figs/stratClass%2.2i.png'%c)
    #plt.xlim(0,30)
    
Yc_train=[]
Yc_valid=[]
for l in kmeans.labels_:
    y=np.zeros((30),int)
    y[l]=1
    Yc_train.append(y)
