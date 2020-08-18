#!/usr/bin/env python
# coding: utf-8

# In[103]:


from keras.datasets import mnist
(train_imgs,train_labels),(test_imgs,test_labels)=mnist.load_data()


# In[104]:


import tensorflow as tf
import matplotlib.pyplot as mplt


# In[105]:


print(train_imgs[1].size,train_imgs[1].shape)

mplt.imshow(train_imgs[1])
mplt.show()


# In[106]:


from keras.utils import to_categorical
train_imgs = train_imgs.astype('float32') / 255
test_imgs = test_imgs.astype('float32') / 255
train_imgs = train_imgs.reshape((60000, 28, 28, 1))
test_imgs = test_imgs.reshape((10000, 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_imgs.shape)
print(test_imgs.shape)


# In[107]:


#dividing the 10000 test images to validation and test sets with 5000 examples in each

val_imgs = test_imgs[:5000]
remain_test_imgs = test_imgs[5000:]

val_labels = test_labels[:5000]
remain_test_labels = test_labels[5000:]


# In[108]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


# In[109]:


model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[110]:


model.summary()


# In[111]:


model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[112]:


model_history=model.fit(train_imgs, train_labels, epochs=5, batch_size=128,validation_data=(val_imgs,val_labels))


# In[113]:


test_loss, test_acc = model.evaluate(remain_test_imgs, remain_test_labels)


# In[ ]:




