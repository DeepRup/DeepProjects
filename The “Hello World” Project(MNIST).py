#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import mnist
(train_imgs,train_labels),(test_imgs,test_labels)=mnist.load_data()


# In[3]:


print(train_imgs.shape)
print(train_labels)


# In[4]:


print(test_imgs.shape)
print(test_labels)


# In[5]:


train_imgs = train_imgs.reshape((60000, 28 * 28))
train_imgs = train_imgs.astype('float32') / 255
test_imgs = test_imgs.reshape((10000, 28 * 28))
test_imgs = test_imgs.astype('float32') / 255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_imgs.shape


# In[5]:


from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[6]:


network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[7]:


network.fit(train_imgs, train_labels, epochs=5, batch_size=128)


# In[ ]:




