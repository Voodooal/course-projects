import os
import numpy as np
import pandas as pd
from surprise import Reader, Dataset
from surprise import KNNBaseline, KNNWithMeans, SVDpp, SVD, CoClustering
from surprise import accuracy
#from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

#%%
os.chdir("/Users/wendymy/Documents/SI671/Homework1")
reader = Reader()
## read the data
tr_data = pd.read_json("reviews.training.json", lines = True)
tr_data = tr_data[['reviewerID', 'asin', 'overall']]
tr_dat = Dataset.load_from_df(tr_data, reader = reader)

#%%
# sample train data for KNN algorithm
sample_ind = np.random.choice(np.arange(tr_data.shape[0]), size = 200000)
sample = tr_data.iloc[sample_ind]
samp_dat = Dataset.load_from_df(sample, reader = reader)

#%%
# development data
dev_data = pd.read_json("reviews.dev.json", lines = True)
dev_data = dev_data[['reviewerID', 'asin', 'overall']]
dev_dat = Dataset.load_from_df(dev_data, reader = reader)

# unlabeled data
te_data = pd.read_csv("reviews.test.unlabeled.csv", header = 0)
te_data = te_data.drop(['datapointID'], axis = 1)
te_data["overall"] = pd.Series(np.zeros(te_data.shape[0]))
te_dat = Dataset.load_from_df(te_data, reader = reader)

#%% 
# define the fit_rmse function
def fit_rmse(algo, data):
    algo.fit(data.build_full_trainset())
    dev_pred = algo.test(dev_dat.build_full_trainset().build_testset())
    dev_rmse = accuracy.rmse(dev_pred, verbose = True)
    tr_rmse = accuracy.rmse(algo.test(data.build_full_trainset().build_testset()), 
                            verbose = True)
    print("rmse on dev_data: " , dev_rmse, "\n",
          "rmse on traning data: ", tr_rmse)

#%%
# define the prediction function 
def prediction(algo):
    algo.fit(tr_dat.build_full_trainset())
    
    preds = list()
    for i in range(te_data.shape[0]):
        uid = te_data['reviewerID'][i]
        iid = te_data['asin'][i]
        preds.append(algo.predict(uid, iid).est)
    return(preds)

#%%
# define output function
def output(algo, filename):
    output = te_data.drop(columns = ['overall'])
    output['overall'] = prediction(algo)
    output.to_csv(filename)
    
#%%
# KNNBaseline
# cross-validation
# sim_options = {'name': 'cosine', 'user_based': False}
# algo = KNNBaseline(k = 10, sim_options = sim_options)
# cross_validate(algo, data, cv = 5, verbose = True)

# use Gridsearch to find the best parameters 
param_grid_knnbl = {'k': list(range(10, 50, 10)),
             'sim_options': {'name':['cosine', 'pearson'],
                            'user_based':[False]}}
gs_knnbl = GridSearchCV(KNNBaseline, param_grid_knnbl, measures = ['rmse','mae'], cv = 5)
gs_knnbl.fit(samp_dat)
algo_knnbl = gs_knnbl.best_estimator['rmse']
print(gs_knnbl.best_score['rmse'])
print(gs_knnbl.best_params['rmse'])

# Use the new parameters with the sampled training data
algo_knnbl = KNNBaseline(k = 20, sim_options = {'name': 'pearson', 'user_based': False})
fit_rmse(algo_knnbl, samp_dat)

output(algo_knnbl, "KNNBaseline_k20_pearson_ii.csv")

#%% 
# SVD
algo_svd = SVD()
fit_rmse(algo_svd, tr_dat)
output(algo_svd, "SVD.csv")

#%%
# SVDpp
algo_svdpp = SVDpp()
fit_rmse(algo_svdpp, tr_dat)    
output(algo_svdpp, "SVDPP.csv")

# Gridsearch of svdpp
param_grid_svdpp = {'lr_all': [0.01, 0.05, 0.08],
             'reg_all': [0.05, 0.1, 0.15]}
gs_svdpp = GridSearchCV(SVDpp, param_grid_svdpp, measures = ['rmse','mae'], cv = 3)
gs_svdpp.fit(tr_dat)
algo_svdpp = gs_svdpp.best_estimator['rmse']
print(gs_svdpp.best_score['rmse'])
print(gs_svdpp.best_params['rmse'])

#%%
# Use the new parameters with the sampled training data
algo_svdpp = SVDpp(lr_all = 0.01, reg_all = 0.15)
fit_rmse(algo_svdpp, tr_dat)
algo_svdpp.fit(tr_dat.build_full_trainset())

#%%
algo_svdpp_new = SVDpp(lr_all = 0.01, reg_all = 0.1)
fit_rmse(algo_svdpp_new, tr_dat)

#%%
output(algo_svdpp, "SVDpp_lr0.01_reg0.15.csv")
output(algo_svdpp_new, "SVDpp_lr0.01_reg0.10.csv")
#%%
# CoClustering
algo_cc = CoClustering()
fit_rmse(algo_cc, tr_dat)
output(algo_cc, "CoClustering.csv")

#%%
# KNNWithMeans
algo_knnwm = KNNWithMeans(k = 40, sim_options = {'name': 'cosine', 'user_based': False})
fit_rmse(algo_knnwm, samp_dat)

# Gridsearch 
param_grid_knnwm = {'k': [30, 40, 50], 
                    'sim_options': {'name':['cosine', 'pearson'],
                                    'user_based':[False]}}
gs_knnwm = GridSearchCV(KNNWithMeans, param_grid_knnwm, measures = ['rmse','mae'], cv = 3)
gs_knnwm.fit(samp_dat)

algo_knnwm = gs_knnwm.best_estimator['rmse']
print(gs_knnwm.best_score['rmse'])
print(gs_knnwm.best_params['rmse'])

# Use the new parameters with the sampled training data
algo_knnwm = KNNWithMeans(k = 30, sim_options = {'name': 'pearson', 'user_based': False})
fit_rmse(algo_knnwm, samp_dat)

#%% 
# error analysis
dev_pred = algo_svdpp.test(dev_dat.build_full_trainset().build_testset())
preds = [i.est for i in dev_pred]
dev_output = te_data.drop(columns = ['overall'])
dev_output['overall'] = preds
dev_output.to_csv("svdpp_dev.csv")

# combine the prediciton with the development overall
dev_output = pd.read_csv("svdpp_dev.csv", header = 0)
dev_output = dev_output.drop(columns = ['index'])
dev_output['original'] = dev_data['overall'].values
dev_output['diff'] = abs(dev_output['overall'].values - dev_output['original'].values)

dev_output.sort_values(by=['diff'], ascending = False)
dev_output[dev_output['diff'] > 2]
tr_data = pd.read_json("reviews.training.json", lines = True)

# take a look at the metadata
tr_data = pd.read_json("reviews.training.json", lines = True)
tr_data[tr_data['reviewerID'] == "A2H1WNB30JNAWU"]
tr_data[tr_data['asin'] == "B00005JKJM"]
np.mean(tr_data[tr_data['reviewerID'] == "A2H1WNB30JNAWU"]['overall'])
np.mean(tr_data[tr_data['asin'] == "B00005JKJM"]['overall'])







