#!/usr/bin/env python
# coding: utf-8

# ### to run this notebook, you need: python 3.7, tensorflow 2.2.0, numpy 1.19.0 or earlier
# #### for the other libraries, the most recent versions are compatible with this notebook

# ## Imports

# In[1]:


# great example here: 
# https://idiotdeveloper.com/dcgan-implementing-deep-convolutional-generative-adversarial-network-in-tensorflow/
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
from tqdm.notebook import tqdm


# ## GPU Support

# In[2]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ## Download Dataset

# In[3]:


# https://drive.google.com/file/d/153i0Xz1osgqKeRG5ImbUUAf2pGMYtRmw/view?usp=sharing
import gdown

url = 'https://drive.google.com/uc?id=153i0Xz1osgqKeRG5ImbUUAf2pGMYtRmw'
output = '../data/phishIRIS_DL_Dataset.zip'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


# In[4]:


from zipfile import ZipFile

with ZipFile(output, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall('../data/')


# In[20]:


def phishIRISTrain(category, num_epochs=2000):
    from pathlib import Path
    import os
    Path('../models').mkdir(parents=True, exist_ok=True)
    Path('../models/saved_model').mkdir(parents=True, exist_ok=True)
    Path('../models/samples').mkdir(parents=True, exist_ok=True)


    # In[21]:


    # hyperparameters
    IMG_H = 64
    IMG_W = 64
    IMG_C = 3
    batch_size = 128
    latent_dim = 128
    # images_path = glob("data/*")
    # category = 'reconnaissance' # either reconnaissance, malware, credencial_phishing, or social _engineering
    images_path = glob(f'../data/phishIRIS_DL_Dataset/train_classified/{category}/*')


    # In[22]:


    # remove models
    try:
        for model in glob('../models/saved_model/*'):
            os.remove(model)
    except OSError:
        pass


    # In[23]:


    w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


    # In[24]:


    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        # resize image
        img = img[:, :, :3]
        return img


    # In[25]:


    def tf_dataset(images_path, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(images_path)
        dataset = dataset.shuffle(buffer_size=10240)
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    # In[26]:


    def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
        x = Conv2DTranspose(
            filters=num_filters,
            kernel_size=kernel_size,
            kernel_initializer=w_init,
            padding="same",
            strides=strides,
            use_bias=False
            )(inputs)

        if bn:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
        return x


    # In[27]:


    def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
        x = Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            kernel_initializer=w_init,
            padding=padding,
            strides=strides,
        )(inputs)

        if activation:
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.3)(x)
        return x


    # In[28]:


    def build_generator(latent_dim):
        f = [2**i for i in range(5)][::-1]
        filters = 32
        output_strides = 16
        h_output = IMG_H // output_strides
        w_output = IMG_W // output_strides

        noise = Input(shape=(latent_dim,), name="generator_noise_input")

        x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((h_output, w_output, 16 * filters))(x)

        for i in range(1, 5):
            x = deconv_block(x,
                num_filters=f[i] * filters,
                kernel_size=5,
                strides=2,
                bn=True
            )

        x = conv_block(x,
            num_filters=3,
            kernel_size=5,
            strides=1,
            activation=False
        )
        fake_output = Activation("tanh")(x)

        return Model(noise, fake_output, name="generator")


    # In[29]:


    def build_discriminator():
        f = [2**i for i in range(4)]
        image_input = Input(shape=(IMG_H, IMG_W, IMG_C))
        x = image_input
        filters = 64
        output_strides = 16
        h_output = IMG_H // output_strides
        w_output = IMG_W // output_strides

        for i in range(0, 4):
            x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

        x = Flatten()(x)
        x = Dense(1)(x)

        return Model(image_input, x, name="discriminator")


    # In[30]:


    class GAN(Model):
        def __init__(self, discriminator, generator, latent_dim):
            super(GAN, self).__init__()
            self.discriminator = discriminator
            self.generator = generator
            self.latent_dim = latent_dim

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super(GAN, self).compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn

        def train_step(self, real_images):
            batch_size = tf.shape(real_images)[0]

            for _ in range(2):
                ## Train the discriminator
                random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                generated_images = self.generator(random_latent_vectors)
                generated_labels = tf.zeros((batch_size, 1))

                with tf.GradientTape() as ftape:
                    predictions = self.discriminator(generated_images)
                    d1_loss = self.loss_fn(generated_labels, predictions)
                grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

                ## Train the discriminator
                labels = tf.ones((batch_size, 1))

                with tf.GradientTape() as rtape:
                    predictions = self.discriminator(real_images)
                    d2_loss = self.loss_fn(labels, predictions)
                grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the generator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            misleading_labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as gtape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = gtape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}


    # In[31]:


    def save_plot(examples, epoch, n):
        examples = (examples + 1) / 2.0
        for i in range(n * n):
            pyplot.subplot(n, n, i+1)
            pyplot.axis("off")
            pyplot.imshow(examples[i])
        filename = f"../models/samples/generated_plot_epoch-{epoch+1}.png"
        pyplot.savefig(filename)
        pyplot.close()


    # In[32]:


    def generate_test_image(model, noise_dim=latent_dim):
        test_input = tf.random.normal([1, noise_dim])
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(5,5))
        plt.imshow((predictions[0, :, :, :] * 127.5 + 127.5) / 255.)
        plt.axis('off') 
        plt.show()


    # In[33]:


    def generate_and_show_images(generator, noise_dim=latent_dim, rows=4, cols=4):
        predictions = generator(tf.random.normal([16, noise_dim]))
        print(predictions.shape)
        fig = plt.figure(figsize=(9,9))
        for i in range(predictions.shape[0]):
            plt.subplot(rows, cols, i+1)
            plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.)
            plt.axis('off') 
            
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


    # In[34]:


    d_model = build_discriminator()
    g_model = build_generator(latent_dim)

    if os.path.exists('../models/saved_model/d_model.h5') and os.path.exists('../models/saved_model/g_model.h5'):
        d_model.load_weights("../models/saved_model/d_model.h5")
        g_model.load_weights("../models/saved_model/g_model.h5")
    else:
        gan = GAN(d_model, g_model, latent_dim)
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        gan.compile(d_optimizer, g_optimizer, bce_loss_fn)
        images_dataset = tf_dataset(images_path, batch_size)

        for epoch in tqdm(range(num_epochs)):
            gan.fit(images_dataset, epochs=1)
            g_model.save("../models/saved_model/g_model.h5")
            d_model.save("../models/saved_model/d_model.h5")

            n_samples = 25
            noise = np.random.normal(size=(n_samples, latent_dim))
            examples = g_model.predict(noise)
            save_plot(examples, epoch, int(np.sqrt(n_samples)))

    d_model.summary()
    g_model.summary()


    # In[ ]:


    generate_test_image(g_model)


    # In[ ]:


    generate_and_show_images(g_model)


    # ## Move generated images to data

    # In[ ]:


    import os 
      
    # importing shutil module 
    import shutil 
      
    from pathlib import Path
    Path('../data/Fake Phish Iris/').mkdir(parents=True, exist_ok=True)


    # In[ ]:


    # Source path 
    source = '../models'
      
    # Destination path 
    destination = f'../data/Fake Phish Iris/models_{category}'
      
    # Move the content of 
    # source to destination 
    dest = shutil.move(source, destination) 


    # In[ ]:




