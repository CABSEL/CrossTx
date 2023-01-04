"""
Created on Sun Dec  5 05:19:22 2021

@author: panagiotischrysinas
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ismember import ismember
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from keras import layers
#from tensorflow.keras import layers
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
#from keras import initializations
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras
import mat4py
import pydot
import pygraphviz
#from keras.utils import plot_model
from mat4py import loadmat
#from keras_tqdm import TQDMNotebookCallback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy import io
from scipy.io import loadmat
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
import pickle as pkl
from scipy.stats.stats import pearsonr   
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from keras.optimizers import adam_v2
#%%
def mean_method(data_mat_nan_pd,union_conc_val,drug_indices_cl_deepce,tgc,num_drugs):

    data_mat_nan_pd1=data_mat_nan_pd.copy()
    data_mat_nan_pd1.pop(tgc)
    pred_final_mat=pd.concat(data_mat_nan_pd1).mean(skipna=True,level=0)
    pred_mean=[None]*num_drugs
    pred_mean_final=[None]*num_drugs
    pred_final_mat_val=pred_final_mat.values
    pred_final_mat_val_tr=pred_final_mat_val.transpose()
    x1=0;
    # i is the drug
    for i in range(0,num_drugs):   
        pred_mean[i]=pred_final_mat_val_tr[:,x1:x1+len(union_conc_val[i])]
        x1=x1+len(union_conc_val[i])

        if len(drug_indices_cl_deepce[tgc][i][0])==0:
            continue
        pred_mean_final[i]=pred_mean[i][:,np.concatenate(drug_indices_cl_deepce[tgc][i][0])-1]
    
        
    return pred_mean_final


def regression_method(ged_src,ged_src_dls,ged_tgc,ged_tgc_dls,tgc, num_genes,num_drugs):
    model_regress=LinearRegression( fit_intercept=False)

    pred_regress=[None]*num_drugs

    non_empty_cells_indices=[i2 for i2,x in enumerate(ged_tgc_dls[tgc]) if  len(x[0])>0]
    #i is the drug that exists in the the target cell line
    for i in non_empty_cells_indices:
        pred_regress[i]=np.zeros((num_genes,len(ged_tgc[tgc][i,1])))

        #dld is the unique drugload
        for dld in range(0,len(ged_tgc[tgc][i,1])):
            #i1 is the gene
            for i1 in range(0,num_genes):
               
                ft = model_regress.fit(ged_src_dls[tgc][i][0], ged_src[tgc][i,i1],sample_weight=None)
                pred_regress[i][i1,dld]=ft.predict([ged_tgc_dls[tgc][i][0][dld]])
   
    return pred_regress

    
def find_nearest_neighbors(prediction_mat,background_data,n1,drug_no):
    
     # n1 is the number of nearest neighbors
    pred_baseline_dl= prediction_mat[drug_no]

    pred_baseline_dlt=pred_baseline_dl.transpose()

    pred_baseline_dlt_mat=pred_baseline_dlt.copy()

    check_dist=[ [0]*background_data.shape[0] for i in range(pred_baseline_dlt_mat.shape[0])]
    ind_check_dist=[ [0]*background_data.shape[0] for i in range(pred_baseline_dlt_mat.shape[0])]
    nn_pca=[]
    for i in range(0,pred_baseline_dlt_mat.shape[0]):
        if not all(np.isfinite(pred_baseline_dlt_mat[i,:])):
                    continue
        for j in range(0,background_data.shape[0]):
            check_dist[i][j]=pearsonr(pred_baseline_dlt_mat[i,:],background_data[j,:])[0]

        ind_check_dist[i]=np.flip(np.argsort(check_dist[i]))                                                 
        nn_pca.append(background_data[ind_check_dist[i][0:n1],:])


    return nn_pca

# PCA projection function
def  pca_projection(prediction_mat,background_data,var_explain_thr,drug_no):
    
    prediction_mat=prediction_mat[drug_no]
    prediction_mat_rnan=prediction_mat[:,~np.all(np.isnan(prediction_mat), axis=0)]
    pca = PCA()
    pca_proj=[]
    
    # j is the drugload
    for j in range(len(prediction_mat_rnan[1,:])):
        
     # apply PCA
     
        pca_bg_data =pca.fit(background_data[j])
        coeffs=pca_bg_data.components_
        coeffs_tr=coeffs.transpose()

        explained_var=pca_bg_data.explained_variance_ratio_.cumsum()
    
        #numb_pcs=np.where(explained_var<var_explain_thr);
        numb_pcs_init=np.where(explained_var>var_explain_thr)

        numb_pcs=np.arange(0,numb_pcs_init[0][0]+1)
        mu=pca_bg_data.mean_                
    
        #scores_pred=np.linalg.pinv(coeffs_tr[:,numb_pcs[0]]).dot(prediction_mat_rnan[:,j]-mu)
        scores_pred=np.linalg.pinv(coeffs_tr[:,numb_pcs]).dot(prediction_mat_rnan[:,j]-mu)
                                            
        #prediction_mat_corr=coeffs_tr[:,numb_pcs[0]].dot(scores_pred)
        prediction_mat_corr=coeffs_tr[:,numb_pcs].dot(scores_pred)

        pca_proj.append(prediction_mat_corr+mu)
    
    return pca_proj

# R2 function for calculating loss during the autoencoder training
def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return  1 - SS_res/(SS_tot + K.epsilon()) 