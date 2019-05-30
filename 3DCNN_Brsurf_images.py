#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os
import glob
import sys
import string
#import matplotlib.pyplot as plt
import cv2

#import seaborn as sns 
#sns.set() 


# In[4]:


################ Load data ###################
def load_data(num_files=2, standard=True):
    os.chdir('/n/scratchlfs/bloxham_lab/ryadav/dipole_files')

    atoz = string.ascii_lowercase
    tags = ['BM+Ra5e5'+letter+str(integer) for integer in [5, 6, 7] for letter in atoz]

    if num_files>0:
        tags = tags[0:num_files]
    else:
        tags = tags

    #read dipole.tag files, a bit messy since data is in fortran 'D' notation
    for ind, item in enumerate(tags):
        print(ind, item)
        fname = 'dipole.'+item
        f = open(fname, 'r')
        data = []
        for k, line in enumerate(f.readlines()):
            st = line.replace('D', 'e')
            data.append(st.split())
        data = np.asarray(data, dtype=float)
        f.close()

        time = data[:,0]
        # dipole tilt angle wrt to the rotation axis
        dip_tilt = data[:,1]
        # convert from [0,180] to [90,-90]
        dip_tilt += -90 

        if ind==0:
            df  = pd.DataFrame({'time':time, 'dipole_tilt':dip_tilt}, dtype=np.float32)
        else:
            df2 = pd.DataFrame({'time':time, 'dipole_tilt':dip_tilt}, dtype=np.float32)

            df = df.append(df2, ignore_index=True)


    # perform standardization 
    if standard==True:
        from sklearn.preprocessing import StandardScaler
        scaled_features = StandardScaler().fit_transform(df[['dipole_tilt']].values)
        #print(scaled_features.shape)
        df['dipole_tilt'] = scaled_features[:,0]
    return df


# In[5]:


############## Equator crossing #################
def get_EqCros(df_main, pad_size=1500, std_thres=0.2):
    
    # get equator crossing times, check 2 points if i is -ve and i+1 is +ve
    # looping over the df is very slow, so construct shifted dataframes 
    df_i = df_main.iloc[0:-1] # first to second last point
    df_iplus1 = df_main.iloc[1:]# second to last point
    
    # multiply i and i+1 values to check -ve values which indicate equator crossing
    df_mul = pd.DataFrame(df_i.values*df_iplus1.values, columns=df_i.columns, index=df_i.index)
    
    df_EquCross  = df_main.loc[df_mul[df_mul['dipole_tilt']<0].index]
    
    # Clean df_EquCross to remove very close data points.
    # First get the indices to drop from the df
    drop_indices=[]
    for i in range(df_EquCross.shape[0]-1):
        #check following 5 points for proximity
        for j in range(5):
            if i+j+1 < df_EquCross.shape[0]: # avoid issue at the end point of df_EquCross
                if df_EquCross.index[i+j+1] - df_EquCross.index[i] < pad_size:
                    drop_indices.append(df_EquCross.index[i+j+1])

    df_EquCross.drop(drop_indices, inplace=True)
    #print(df_EquCross.shape)
    
    # Add new columns containing the nature and standard deviation of dipole 
    # tilt in the left and right paddings of the equator crossing times
    df_EquCross['nature'] = pd.Series(np.zeros(df_EquCross.shape[0]), 
                                      index=df_EquCross.index, 
                                      dtype=int)
    df_EquCross['left_std'] = pd.Series(np.zeros(df_EquCross.shape[0]), 
                                      index=df_EquCross.index, 
                                      dtype=float)
    df_EquCross['right_std'] = pd.Series(np.zeros(df_EquCross.shape[0]), 
                                      index=df_EquCross.index, 
                                      dtype=float)
    
    # get nature of equtor crossings
    for i in range(df_EquCross.shape[0]): 
        LeftInds = [df_EquCross.index[i]-j for j in range(pad_size)]
        LeftSum = df_main.iloc[LeftInds]['dipole_tilt'].sum()/pad_size
        # access elements by '.at' function -> df.at[index, column]
        df_EquCross.at[df_EquCross.index[i], 'left_std'] = df_main.iloc[LeftInds]['dipole_tilt'].std()

        # Summing on the right hand side padding may cause issues if the last 
        # equator crossing is being considered.
        # So, shrink the padding size equal to the no. of last available points
        # at the end of the series
        if i == df_EquCross.shape[0] - 1: # basically, last iteration
            pad_size_temp = df_main.index[-1] - df_EquCross.index[i]
        else:
            pad_size_temp = pad_size

        RightInds = [df_EquCross.index[i]+j for j in range(pad_size_temp)]
        RightSum = df_main.iloc[RightInds]['dipole_tilt'].sum()/pad_size_temp
        df_EquCross.at[df_EquCross.index[i], 'right_std'] = df_main.iloc[RightInds]['dipole_tilt'].std()

        if LeftSum*RightSum > 0:
            # access elements by '.at' function -> df.at[index, column]
            df_EquCross.at[df_EquCross.index[i], 'nature'] = 0
        else:
            df_EquCross.at[df_EquCross.index[i], 'nature'] = 1

    # use std threshold for selecting cleaner reversals/excursions
    cond2 = (df_EquCross['left_std']<std_thres) & (df_EquCross['right_std']<std_thres)
    df_EquCross_clean = df_EquCross[cond2]
    
    return df_EquCross_clean


# In[6]:


############## frame reader ##################
def read_frames(global_ind=1000, num_frames=100):
    os.chdir('/n/scratchlfs/bloxham_lab/ryadav/BrCMB_Res1')
    filenames = ['brcmb_{:010}.png'.format(global_ind-i) for i in range(num_frames) ]
    image_list = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in filenames]
    image_list = np.asarray(image_list)
    if len(image_list.shape)==1:
        print('Image not found.')
    return image_list


# In[7]:


# dipole file data dir
os.chdir('/n/scratchlfs/bloxham_lab/ryadav/dipole_files')

df_main = load_data(num_files=-1)

# Read lookup table
# colums arrangement:
# BM+Ra5e5+tag         a5
# time            11037.1
# ind_in_file        1000
lookup_table = pd.read_csv('lookup_table.csv')


# In[8]:


# padding size (in no. of points) of the analysis window around the equator crossings
pad_size = 1000
std_thres = 0.3 # North hemi has about 0 to 1 normalized latitude range
create_frames=False

# get where reversal/excursions happened
df_EquCross_forward = get_EqCros(df_main, pad_size=pad_size, std_thres=std_thres)
print(df_EquCross_forward.info())

# plot number of reversals/excursions
#fig, ax = plt.subplots(figsize=(7,5))
#sns.countplot(df_EquCross_forward['nature'])
#plt.show()


# In[9]:


#---------------plot few events
if False:
    fig_rows=5
    fig_cols=3
    plot_window_size=pad_size

    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(12,15))
    fig.tight_layout()

    for i in range(fig_rows):
        for j in range(fig_cols):
            rand_ind = np.random.randint(0,df_EquCross_forward.shape[0])
            LeftInds = [df_EquCross_forward.index[rand_ind]-k for k in range(plot_window_size)]
            ax[i,j].plot( df_main.iloc[LeftInds]['dipole_tilt'], '-', color='royalblue', lw=3)

            RightInds = [df_EquCross_forward.index[rand_ind]+k for k in range(plot_window_size)]
            ax[i,j].plot( df_main.iloc[RightInds]['dipole_tilt'], '-', color='coral', lw=3)
            
            ax[i,j].set_ylim([-2,2])

            if df_EquCross_forward.at[df_EquCross_forward.index[rand_ind], 'nature']==0:
                ax[i,j].set_title('Excursion')
            else:
                ax[i,j].set_title('Reversal')
            #ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].axes.get_xaxis().set_ticks([])
            if j==0:
                ax[i,j].set_ylabel('Dipole tilt')
            if i==fig_rows-1:
                ax[i,j].set_xlabel('Time')
                
    plt.show()


# In[43]:


from sklearn.model_selection import train_test_split

num_events = -1
num_frames=10

X, y = [], []
for ind, event in enumerate(df_EquCross_forward.index[0:num_events]):
    chunk = read_frames(global_ind=event, num_frames=num_frames)
    print(event)
    if len(chunk.shape)!=1:# avoid data without frames
        X.append(chunk)
        y.append(df_EquCross_forward['nature'][event])
        

X = np.asarray(X)
y = np.asarray(y)

X = X.reshape((X.shape[0], 
               X.shape[1], 
               X.shape[2], 
               X.shape[3], 
               1))
print(X.shape)
print(y.shape)


# Train test spilling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print('X train={}, Y train={}.'.format(X_train.shape, y_train.shape))
print('X test={}, Y test={}'.format(X_test.shape, y_test.shape))


# In[45]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

kernel_size = 5
epochs=4
batch_size=32

model = Sequential()
# 1st layer group
model.add(Convolution3D(16, (1, kernel_size, kernel_size), activation='relu',  
                        input_shape=(X_train.shape[1:])))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Convolution3D(32, (1, kernel_size, kernel_size), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Convolution3D(32, (1, kernel_size, kernel_size), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Convolution3D(32, (1, kernel_size-1, kernel_size-1), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Convolution3D(32, (1, kernel_size-1, kernel_size-1), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Flatten())

#model.add(Dense(4096, activation='relu'))
#model.add(Dropout(.5))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size=batch_size,
                  validation_data=(X_test, y_test),
                  )


# get ROC curve for the model
from sklearn.metrics import roc_curve, auc
# ------------get ROC curve for the model
y_test_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area under curve = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')


fig, ax = plt.subplots(figsize=(15,5))
plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')



plt.show()
