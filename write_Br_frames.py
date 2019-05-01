import numpy as np
import pandas as pd
import os
import glob
import sys
import string
import matplotlib.pyplot as plt
import cv2

import seaborn as sns 
sns.set() 

################ Load data ###################
def load_data(num_files=2, standard=True):
    os.chdir('/home/ryadav/Meduri_reversals/dipole_files')

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
        X = []
        for k, line in enumerate(f.readlines()):
            st = line.replace('D', 'e')
            X.append(st.split())
        X = np.asarray(X, dtype=float)
        f.close()

        time = X[:,0]
        # dipole tilt angle wrt to the rotation axis
        dip_tilt = X[:,1]
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


############## frame generator ##################
def write_frames(lookup_table, global_ind=1000, num_frames=100):
    
    status = []
    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    os.chdir('/home/ryadav/Meduri_reversals/')
    all_global_inds = [global_ind-i for i in range(num_frames)] #decreases by index
    unique_tags = lookup_table.loc[all_global_inds]['BM+Ra5e5+tag'].unique()
    if len(unique_tags)==1: # skip event spanning multiple files, not many
        tag = lookup_table.loc[global_ind]['BM+Ra5e5+tag']
        time = lookup_table[lookup_table['BM+Ra5e5+tag']==tag]['time']
        local_step_ind = lookup_table.loc[global_ind]['ind_in_file']
        # get local indices in the corresponding dipole file
        ks = [local_step_ind-i for i in range(num_frames)] #decreases by index
        
        # Br cmb section
        cmb = MagicCoeffCmb(tag='BM+Ra5e5'+tag, iplot=False)
        interp_blm = np.zeros((len(time), cmb.blm.shape[1]), dtype=np.complex64)
        #print(interp_blm.shape)
        for k in range(cmb.blm.shape[1]):
            interp_blm[:,k] = np.interp(time, cmb.time, cmb.blm[:,k])

        nlat = int(max(int(cmb.l_max_cmb*(3./2./2.)*2.),192))
        nphi = int(2*nlat/cmb.minc)

        # Define spectral transform setup
        sh = SpectralTransforms(l_max=cmb.l_max_cmb, minc=cmb.minc,
                                lm_max=cmb.lm_max_cmb,
                                n_theta_max=nlat)
        # plot figure
        fname_ind = 0
        for k in ks:
            #print(k)
            BrCMB = sh.spec_spat(interp_blm[k, :]*cmb.ell*(sh.ell+1)/cmb.rcmb**2)
            ax.imshow(BrCMB.T, cmap='gray', aspect='auto', 
                      vmin=-2*np.std(BrCMB), 
                      vmax= 2*np.std(BrCMB)
                      )
            ax.set_axis_off()
            fname = 'BrCMB_Res1/brcmb_{:010}.png'.format(global_ind-fname_ind)
            fig.savefig(fname, dpi=50)
            fname_ind+=1
    else:
        status=[global_ind]
        
    return status



# dipole file data dir
os.chdir('/home/ryadav/Meduri_reversals/dipole_files')

df_main = load_data(num_files=-1)

# Read lookup table
# colums arrangement:
# BM+Ra5e5+tag         a5
# time            11037.1
# ind_in_file        1000
lookup_table = pd.read_csv('lookup_table.csv')

# padding size (in no. of points) of the analysis window around the equator crossings
pad_size = 1000
std_thres = 0.3 # North hemi has about 0 to 1 normalized latitude range
create_frames=False

# get where reversal/excursions happened
df_EquCross_forward = get_EqCros(df_main, pad_size=pad_size, std_thres=std_thres)

# plot number of reversals/excursions
#fig, ax = plt.subplots(figsize=(7,5))
#sns.countplot(df_EquCross_forward['nature'])
#plt.show()

if create_frames==True:
    counter=1
    skipped_global_inds = []
    for i in df_EquCross_forward.index:
        status=write_frames(lookup_table, global_ind=i, num_frames=200)
        if len(status)!=0:
            print('Skipping event spanning multiple files.')
            skipped_global_inds.append(status[0])
        else:
            print('{} of {} is done.'.format(counter, len(df_EquCross_forward.index)))
        counter+=1




