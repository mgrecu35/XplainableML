import tensorflow as tf
import tensorflow.keras as keras
codings_size=4+18
generator = keras.models.Sequential([
    keras.layers.Dense(20, activation="relu", input_shape=[codings_size]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(36, activation="linear"),
])
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[36]),
    keras.layers.Dense(20, activation="selu"),
    keras.layers.Dense(20, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

from netCDF4 import Dataset
fh=Dataset("trainingData.nc")
tData=fh["tData"][:,:,0:2]
batch_size = 32

tData[tData<-10]=-10
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
                tDataS[:,i,j]=(tData[:,i,j]-tData_m[0,j])/tData_s[0,j]


scale(tData,tDataS,tData_m,tData_s)

X_train=tDataS[:,35::-1,0]
x_train=tDataS[:,36:18:-1,0]
import numpy as np
ind=np.arange(tDataS.shape[0])
dataset = tf.data.Dataset.from_tensor_slices(ind).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

#stop

discriminator.compile(loss="binary_crossentropy", optimizer="Adam")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="Adam")

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=5):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs)) 
        for ind in dataset:
            # phase 1 - training the discriminator
            noise = np.random.randn(batch_size, codings_size)
            noise[:,0:18]=x_train[ind,:]
            noise=tf.constant(noise)
            generated_images = generator(noise)
            X_batch=tf.constant(X_train[ind,:].astype(np.float32))
            #print(generated_images.dtype)
            #print(X_batch.dtype)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)


train_gan(gan, dataset, batch_size, codings_size, n_epochs=15)

generator, discriminator = gan.layers

n=x_train.shape[0]

noise = np.random.randn(n, codings_size)
noise[:,0:18]=x_train[:,:]
noise=tf.constant(noise)
y=generator(noise)
