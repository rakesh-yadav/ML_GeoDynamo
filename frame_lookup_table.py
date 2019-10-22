import numpy as np
import pandas as pd
import os
import glob
import sys
import string

# creat a data frame with tag names, global time, 
# time step index within dipole file, and a global lookup index

os.chdir('/home/ryadav/Meduri_reversals/dipole_files')

atoz = string.ascii_lowercase
tags = [letter+str(integer) for integer in [5, 6, 7] for letter in atoz]

#num_files = -1
#tags = tags[0:num_files]

dtypes =   {
            'BM+Ra5e5+tag':str,
            'time':np.float64,
            'ind_in_file':np.int
            }

for ind, item in enumerate(tags):
    # read time from dipole.tag files
    print(ind, item)
    fname = 'dipole.BM+Ra5e5'+item
    f = open(fname, 'r')
    X = []
    for k, line in enumerate(f.readlines()):
        st = line.replace('D', 'e')
        X.append(st.split())
    X = np.asarray(X, dtype=float)
    f.close()

    time = X[:,0]
    tag_names = [item for i in range(len(time))]
    time_step_inds = np.arange(len(time))

    if ind==0:
        df  = pd.DataFrame({
                            'BM+Ra5e5+tag':tag_names,
                            'time':time,
                            'ind_in_file':time_step_inds
                            } )
    else:
        df2  = pd.DataFrame({
                            'BM+Ra5e5+tag':tag_names,
                            'time':time,
                            'ind_in_file':time_step_inds
                            } )
        df = df.append(df2, ignore_index=True)
        
df = df.astype(dtypes)
df.to_csv('lookup_table.csv', index=False)
