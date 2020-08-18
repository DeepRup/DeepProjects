#!/usr/bin/env python
# coding: utf-8

# In[116]:


import tensorflow as tf
import matplotlib.pyplot as mplt


# In[117]:


(train_imgs,train_labels),(test_imgs,test_labels)=tf.keras.datasets.fashion_mnist.load_data()

print("train_image shape:", train_imgs.shape, "train_labels shape:", train_labels.shape)
print("test_image shape:", test_imgs.shape, "test_labels shape:", test_labels.shape)


# In[118]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[119]:


print(train_imgs[1].size,train_imgs[1].shape)

mplt.imshow(train_imgs[1])
mplt.colorbar()
mplt.grid(False)
mplt.show()


# In[120]:


from keras.utils import to_categorical
train_imgs = train_imgs.reshape((60000, 28, 28, 1))
test_imgs = test_imgs.reshape((10000, 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_imgs = train_imgs.astype('float32') / 255
test_imgs = test_imgs.astype('float32') / 255

print(train_imgs.shape)
print(test_imgs.shape)


# 

# In[136]:


#dividing the 10000 test images to validation and test sets with 5000 examples in each

val_imgs = test_imgs[:5000]
remain_test_imgs = test_imgs[5000:]

val_labels = test_labels[:5000]
remain_test_labels = test_labels[5000:]


# In[122]:


model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()


# 
# 
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dropout(0.5))

# In[123]:


model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[124]:


model_history=model.fit(train_imgs, train_labels, epochs=10, batch_size=128,validation_data=(val_imgs,val_labels))


# In[125]:


# Evaluate the model on test set
score = model.evaluate(remain_test_imgs, remain_test_labels)


# In[148]:


predicted_test_imgs=model.predict(remain_test_imgs)
print(predicted_test_imgs)


# mplt.figure(figsize=(10,10))
# for i in range(25):
#     mplt.subplot(5,5,i+1)
#     mplt.xticks([])
#     mplt.yticks([])
#     mplt.grid(False)
#     mplt.imshow(remain_test_imgs[i])
#     if(predicted_test_imgs[i] == remain_test_imgs[i]):
#      mplt.xlabel(class_names[remain_test_labels[i]])
# mplt.show()

# In[ ]:




