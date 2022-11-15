#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import os

import numpy as np
import matplotlib.pyplot as plt

print("tensorflow version: ", tf.__version__)
print("numpy version:",np.__version__)


# In[2]:


dataset_dir = os.path.join(os.getcwd(), )

dataset_treino_dir = os.path.join(dataset_dir, 'dataset_treino')
dataset_treino_fogo_len = len(os.listdir(os.path.join(dataset_treino_dir, 'fogo')))
dataset_treino_floresta_len = len(os.listdir(os.path.join(dataset_treino_dir, 'floresta',)))


dataset_validation_dir = os.path.join(dataset_dir, 'dataset_validation')
dataset_validation_fogo_len = len(os.listdir(os.path.join(dataset_validation_dir, 'fogo')))
dataset_validation_floresta_len = len(os.listdir(os.path.join(dataset_validation_dir, 'floresta',)))

print('Treino imagens floresta pegando fogo:%s' % dataset_treino_fogo_len)
print('Treino imagens floresta sem fogo:%s' % dataset_treino_floresta_len)
print('Validação imagens floresta pegando fogo:%s' % dataset_validation_fogo_len)
print('Validação imagens floresta pegando fogo:%s' % dataset_validation_fogo_len)


# In[3]:


imagem_largura, imagem_altura = 160, 160
imagem_cor_canal = 3
imagem_cor_canal_tamanho = 255
imagem_tamanho = (imagem_largura, imagem_altura)
imagem_shape = imagem_tamanho + (imagem_cor_canal,)

batch_tamanho = 32 # informações por vez que tirarei do meu dataset
epochs = 20 # numero de vezes que passarei na imagem inteira
taxa_de_aprendizagem = 0.0001 #

classificacao = ['floresta_sem_queimada', 'floresta_em_queimada']


# In[4]:


dataset_treino = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_treino_dir,
    image_size = imagem_tamanho,
    batch_size = batch_tamanho,
    shuffle = True)


dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size = imagem_tamanho,
    batch_size = batch_tamanho,
    shuffle = True
)


# In[5]:


dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5

dataset_test = dataset_validation.take(dataset_validation_batches)
dataset_validation = dataset_validation.skip(dataset_validation_batches)

print("Dados de validação cardinalidade: %d " % tf.data.experimental.cardinality(dataset_validation))
print("Dados de teste cardinalidade %d" % tf.data.experimental.cardinality(dataset_test))


# In[6]:


def exibir_dataset(dataset):
    plt.gcf().clear()
    plt.figure(figsize = (15, 15))
    for info, rotulo in dataset.take(1):
        for i in range(9):
            plt.subplot(4, 4, i+1)
            plt.axis('off')
            plt.imshow(info[i].numpy().astype('uint8'))
            plt.title(classificacao[rotulo[i]])


# In[7]:


exibir_dataset(dataset_treino)


# In[8]:


exibir_dataset(dataset_validation)


# In[9]:


exibir_dataset(dataset_test)


# In[10]:


modelo = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling( 1. / imagem_cor_canal_tamanho, input_shape = imagem_shape),
    tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])


# In[11]:


modelo.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = taxa_de_aprendizagem),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)


# In[12]:


historia = modelo.fit(
    dataset_treino,
    validation_data = dataset_validation,
    epochs = epochs
)


# In[13]:


def exibir_dataset_predicoes(dataset):
    
    info, catalogo = dataset.as_numpy_iterator().next()
    
    predicoes = modelo.predict_on_batch(info).flatten()
    predicoes = tf.where(predicoes < 0.5, 0, 1)
    
    print('Catalogos:      %s' % catalogo)
    print('Predicoes: %s' % predicoes.numpy())
    
    plt.gcf().clear()
    plt.figure(figsize = (15, 15))
    
    for i in range(9):
        
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        
        plt.imshow(info[i].astype('uint8'))
        plt.title(classificacao[predicoes[i]])


# In[14]:


exibir_dataset_predicoes(dataset_test)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




