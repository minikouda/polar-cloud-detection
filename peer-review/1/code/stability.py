'''
Following the pipeline outlined in stability.ipynb

Will intake npz filepaths and output new perturbed npz files 
denoted as <image>_<perturbation>.npz'''

import pandas as pd
import numpy as np
import os 

def gauss_noise(
        filepaths, 
        features,           # features to perturb
        labeled=False,      # difference in labeled vs unalabeled data
        mu=0, 
        sigma=0.1
        ): 
    
    # extract filenames
    names = [os.path.splitext(os.path.basename(fp))[0] for fp in filepaths]
    directory = os.path.dirname(filepaths[0])       # assuming all files from same directory
    
    # load images
    dfs = []
    for filepath in filepaths: 
        with np.load(filepath) as data: 
            target_array = data['arr_0']
        
        if labeled: 
            temp = pd.DataFrame(target_array[:, 0:11], 
                                columns = ['y', 'x', 'ndai', 'sd', 'corr', 
                                'df', 'cf', 'bf', 'af', 'an', 'label'])
            dfs.append(temp)
        else: 
            temp = pd.DataFrame(target_array[:, 0:10], 
                                columns = ['y', 'x', 'ndai', 'sd', 'corr', 
                                'df', 'cf', 'bf', 'af', 'an'])
            dfs.append(temp)

    # perturb images
    perturbed_dfs = []
    for df in dfs: 
        pert_temp = _gauss_noise(df, features, mu=mu, sigma=sigma)
        perturbed_dfs.append(pert_temp)


    # put new files back in data folder with new names 
    i = 0
    # sanity check
    if len(names) == len(perturbed_dfs): 
        while i < len(names): 
            output_path = f"{directory}/{names[i]}_gauss.npz"
            target_array = perturbed_dfs[i].values
            np.savez_compressed(output_path, arr_0=target_array)
            i = i + 1            

    return
    

    
# gauss_noise helper function    
def _gauss_noise(df, features, mu=0, sigma=0.1): 
    df_noisy = df.copy()
    for feat in features: 
        df_noisy[feat] = df_noisy[feat] + np.random.normal(mu, sigma, size=len(df_noisy))
    return df_noisy