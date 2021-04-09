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
XpRate=pRate.copy()
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
        if pType[i]>=1 and (bzd[i]-136)*(bzd[i]-140)<0:
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

def prepareData(nz1,nz2,X,Y,bzd,batch_size):
    scalerX=StandardScaler()
    scalerX2=StandardScaler()
    scalerY=StandardScaler()
    a=np.nonzero((bzd-136)*(bzd-140)<0)
    X_sub=scalerX.fit_transform(X[a[0],:nz1])
    Y_sub=scalerY.fit_transform(Y[a[0],:nz2])
    n=X_sub.shape[0]
    r=np.random.random(n)
    a=np.nonzero(r>0.25)
    b=np.nonzero(r<0.25)
    X_train=X_sub[a[0],:nz1].copy()
    X_valid=X_sub[b[0],:nz1].copy()
    Y_train=Y_sub[a[0],:nz2].copy()
    Y_valid=Y_sub[b[0],:nz2].copy()
    ind=np.arange(X_train.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices(ind).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    return X_train,X_valid,Y_train, Y_valid, dataset

batch_size=128
X_train,X_valid,Y_train, Y_valid, datasetS=prepareData(68,34,XKa,XpRate,bzd,batch_size)

from ganDef import *


ninput=68
coding_size=6
nout=34
gen2=generator_2(ninput,coding_size,nout)
disc2=discriminator_2(ninput,nout)
gan_DPR=gan2(ninput,coding_size,nout)

gan_DPR.compile(loss="binary_crossentropy", optimizer="Adam")

n_epochs=500
train_gan2(gan_DPR, datasetS, batch_size, coding_size, X_train,Y_train,n_epochs)


gen2, disc = gan_DPR.layers[2:]

n=X_valid.shape[0]
inp1=tf.constant(X_valid[:,:])
noise = tf.random.normal(shape=[n, coding_size])
y=gen2([inp1,noise])

for i in range(-10,0):
  print(np.corrcoef(y[:,i],Y_valid[:,i]))
