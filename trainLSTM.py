import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np 
from netCDF4 import Dataset

print('reading training data!')
print('still reading...')
fh=Dataset("trainingData.nc")
tData=fh["tData"][:,:,:]
#for i in range(2,5):
#    tData[:,:,i]=np.log(tData[:,:,i]+1e-5)
tData[:,:,0:2][tData[:,:,0:2]<0]=0

for i in range(tData.shape[0]):
    tData[i,:,2]=tData[i,:,2][::-1].cumsum()[::-1]
    tData[i,:,3]=tData[i,:,3][::-1].cumsum()[::-1]
    
tData_m=tData.mean(axis=0)
tData_s=tData.std(axis=0)
tDataS=tData.copy()*0

from numba import jit

@jit(nopython=True)
def scale(tData,tDataS,tData_m,tData_s):
    nx,ny=tData_m.shape
    print(nx,ny)
    for i in range(nx):
        for j in range(ny):
            if(tData_s[i,j]>0):
                tDataS[:,i,j]=(tData[:,i,j]-tData_m[i,j])/tData_s[i,j]

scale(tData,tDataS,tData_m,tData_s)
n=tData.shape[0]
r=np.random.random(n)
a=np.nonzero(r>0.15)
b=np.nonzero(r<0.15)
x_train=tDataS[a[0],50::-1,4:]
y_train=tDataS[a[0],50::-1,0:2]

x_val=tDataS[b[0],50::-1,4:]
y_val=tDataS[b[0],50::-1,0:2]

print('reading done')

def lstm_model(ndims=7):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.GRU(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.GRU(6, recurrent_activation='sigmoid',return_sequences=True)(out1)
    #out1 = tf.keras.layers.LSTM(6, return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(2, return_sequences=True)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

model=lstm_model(7)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),  \
    loss='mse',\
    metrics=[tf.keras.metrics.MeanSquaredError()])


history = model.fit(x_train, y_train, batch_size=32,epochs=100,
                    validation_data=(x_val, y_val))
