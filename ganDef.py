import tensorflow as tf
import tensorflow.keras as keras
codings_size=4
ninput=36
import numpy as np
def generator_2(ninput,coding_size,nout):
    inp1 = tf.keras.layers.Input(shape=(ninput,))
    inp2 = tf.keras.layers.Input(shape=(coding_size,))
    inp = tf.concat([inp1,inp2],axis=-1)
    out1 = tf.keras.layers.Dense(50,activation="relu")(inp)
    out1 = tf.keras.layers.Dense(50,activation="relu")(out1)
    out = tf.keras.layers.Dense(nout)(out1)
    model = tf.keras.Model(inputs=[inp1,inp2], outputs=out)
    return model

def discriminator_2(ninput,nout):
    inp1 = tf.keras.layers.Input(shape=(ninput,))
    inp2 = tf.keras.layers.Input(shape=(nout,))
    inp = tf.concat([inp1,inp2],axis=-1)
    out1 = tf.keras.layers.Dense(50,activation="selu")(inp)
    out1 = tf.keras.layers.Dense(50,activation="selu")(out1)
    out = tf.keras.layers.Dense(1,activation="sigmoid")(out1)
    model = tf.keras.Model(inputs=[inp1,inp2], outputs=out)
    return model

def gan2(ninput,coding_size,nout):
    inp1 = tf.keras.layers.Input(shape=(ninput,))
    inp2 = tf.keras.layers.Input(shape=(coding_size,))
    out1=generator_2(ninput,coding_size,nout)([inp1,inp2])
    disc2=discriminator_2(ninput,nout)
    disc2.compile(loss="binary_crossentropy", optimizer="Adam")
    disc2.trainable = False
    out=disc2([inp1,out1])
    model = tf.keras.Model(inputs=[inp1,inp2], outputs=out)
    return model


def train_gan2(gan2m, dataset, batch_size, codings_size, X_train,Y_train,n_epochs=5):
    generator_2, discriminator_2 = gan2m.layers[2:]
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs)) 
        for ind in dataset:
            # phase 1 - training the discriminator
            noise = np.random.randn(batch_size, codings_size)
            inp1=tf.constant(X_train[ind,:])
            noise=tf.constant(noise)
            generated_images = generator_2([inp1,noise])
            Y_batch=tf.constant(Y_train[ind,:].astype(np.float32))
            X_fake_and_real = tf.concat([inp1,inp1], axis=0)
            Y_fake_and_real = tf.concat([generated_images,Y_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator_2.trainable = True
            discriminator_2.train_on_batch([X_fake_and_real,Y_fake_and_real], y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator_2.trainable = False
            gan2m.train_on_batch([inp1,noise], y2)
