
# Copyright (C) 2022 Yegor Tkachenko, Wassim Dhaouadi, Kamel Jedidi
# Code -- Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction
# https://github.com/computationalmarketing/scalable_ranking_under_constraints/

# Code running ranking of 50 movies


from core_functions_unbalanced import *

import numpy as np
import pandas as pd
import cvxopt
from scipy.optimize import linear_sum_assignment
import time
import cvxpy as cp
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
import math
import json
import seaborn as sns
sns.set_theme(style="whitegrid")
import os

if not os.path.exists('../results'):
    os.makedirs('../results')

PATH_RESULTS = '../results/ml-25m-50'
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)


# load data
PATH_DATA = '../data/ml-25m'
ratings = pd.read_csv(PATH_DATA + '/generated_data.csv')
ratings.shape

# very important to sort
# we can use the special structure of the problem to speed up computation
ratings = ratings.sort_values('rating', ascending=False)

# number of unique movies
ratings['movieId'].unique().shape

# dummy-code user clusters
ratings = pd.get_dummies(ratings,columns=['userClust'],drop_first=True)

def extract_data(user, top_k, sample_size):
    # data extraction function
    # user is user id in range(1000)
    # sample - whether to extract sample prop of user observations only
    # top_k - across what items to measure the utility/exposure

    # user data
    ratings_u = ratings[ratings['userId']==user]
    
    # for each user, optimize only across top sample_size items 
    if sample_size:
        ratings_u = ratings_u.iloc[:sample_size]

    n = ratings_u.shape[0]

    # utilities and constraints
    exposure = np.array([[1.0/np.log2(i+1.0) for i in range(1,top_k+1)]])
    
    # taking dot product with identical discounting
    U = np.dot(ratings_u['rating'].values[:,np.newaxis], exposure)
    
    A_queer = np.dot(ratings_u['queer_score'].values[:,np.newaxis], exposure)
    A_race = np.dot(ratings_u['race_score'].values[:,np.newaxis], exposure)
    A_free_speech = np.dot(ratings_u['free_speech_score'].values[:,np.newaxis], exposure)
    A_scifi = np.dot(ratings_u['scifi_score'].values[:,np.newaxis], exposure)
    A_new = np.dot(ratings_u['year'].values[:,np.newaxis]-1990, exposure)/top_k/100
    
    # constraint A_k list
    A_list = [A_queer, A_race, A_free_speech, A_scifi, A_new]
    
    # b_k list
    b = np.sum(exposure)*np.array([0.1, 0.1, 0.1, 0.1, 0.0])

    # user covariates
    X = ratings_u[['e{}'.format(i) for i in range(25)]].iloc[0]
    
    return U, A_list, b, X



lambda_names = ['Queer boost', 'Race boost', 'Free speech boost', 'Sci-fi boost', 'Newsiness boost'] #  'Police boost', 
n_constraints = len(lambda_names)

X_names = ['e'+str(i) for i in range(25)] #+ ['userClust_{}'.format(i) for i in range(1,10)]

train_size = 750
N_USERS = 1000


METHOD = 'rearrangement'
TOP_K = 50
SAMPLE_SIZE = 1000


np.random.seed(999)

users_train = np.random.choice(range(N_USERS), size=train_size, replace=False)
users_test = np.setdiff1d(np.arange(N_USERS),users_train)


### Finetuning EPSILON to break ties
eps = [1.0] + [1. + i*10**(-j) for j in range(1,5) for i in range(1,10)]
eps_constraint_violation = []
for u in tqdm(users_train[:10]):
    U, A_list, b, X = extract_data(u, top_k=TOP_K, sample_size=SAMPLE_SIZE)
    lmbd = user_dual(U, A_list, b)
    for e in eps:
        P, S = optimal_ranking(U, A_list, lmbd, method=METHOD, EPSILON=e)
        mcv = np.mean([1*(np.trace(A_list[i].T.dot(P))<b[i]) for i in range(len(A_list))])
        eps_constraint_violation.append({'u':u,'e':e, 'mcv':mcv})

eps_constraint_violation = pd.DataFrame(eps_constraint_violation)

eps_constraint_violation = eps_constraint_violation.groupby('e').mean().reset_index().sort_values('e').reset_index(drop=True)
ind_opt_eps = eps_constraint_violation.index[eps_constraint_violation['mcv']==eps_constraint_violation['mcv'].min()].tolist()
if len(ind_opt_eps)>0:
    EPSILON = eps_constraint_violation.loc[ind_opt_eps[0]]['e'] # breaking ties
else:
    EPSILON = 1.0001


# NO ETHICAL ADJUSTMENT
print('running no adjustment')

constraint_violations_no_optim = []
utility_no_optim = []
time_no_optim = []

np.random.seed(999)

for u in tqdm(users_test):

    U, A_list, b, X = extract_data(u, top_k=TOP_K, sample_size=SAMPLE_SIZE)

    start = time.time()
    P = np.eye(U.shape[0])
    end = time.time()
    
    constraint_violations_u = [1*(np.trace(A_list[i].T.dot(P))<b[i]) for i in range(len(A_list))]
    constraint_violations_no_optim.append(constraint_violations_u)
    utility_no_optim.append(np.trace(U.T.dot(P)))
    time_no_optim.append(1000*(end - start))



# OPTIMAL LAMBDA
print('running optimal lambda')

constraint_violations_lambda_optimal = []
utility_lambda_optimal = []
time_lambda_optimal = []

np.random.seed(999)

lambda_all = []
X_all = []
for u in tqdm(range(N_USERS)):

    U, A_list, b, X = extract_data(u, top_k=TOP_K, sample_size=SAMPLE_SIZE)

    start = time.time()
    lmbd = user_dual(U, A_list, b)
    P, S = optimal_ranking(U, A_list, lmbd, method=METHOD,EPSILON=EPSILON)
    end = time.time()

    lambda_all.append(lmbd)
    X_all.append(X)

    if u in users_test:
        constraint_violations_u = [1*(np.trace(A_list[i].T.dot(P))<b[i]) for i in range(len(A_list))]
        constraint_violations_lambda_optimal.append(constraint_violations_u)
        utility_lambda_optimal.append(np.trace(U.T.dot(P)))
        time_lambda_optimal.append(1000*(end - start))

lambda_all = pd.DataFrame(np.array(lambda_all).squeeze())
lambda_all.columns = lambda_names
lambda_all.to_csv(PATH_RESULTS+ '/lambda_all.csv',index=False)

X_all = pd.DataFrame(np.array(X_all).squeeze())
X_all.columns = X_names
X_all.to_csv(PATH_RESULTS+ '/X_all.csv',index=False)


lambda_all = pd.read_csv(PATH_RESULTS+ '/lambda_all.csv')



plt.figure(figsize=(10,8))
gridspec.GridSpec(4,1)

temp = lambda_all.stack().reset_index()
temp.columns = ['ind', 'Constraint', 'Lambda'] 

plt.subplot2grid((4,1), (0,0), rowspan=3)
ax = sns.boxplot(x='Lambda', y='Constraint', data=temp[temp['Constraint']!='Newsiness boost'], whis=[0,100])
ax.xaxis.set_ticks_position('top')
plt.xlabel('',fontsize=24)
plt.ylabel('',fontsize=24)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)

plt.subplot2grid((4,1), (3,0))
ax = sns.boxplot(x='Lambda', y='Constraint', data=temp[temp['Constraint']=='Newsiness boost'], whis=[0,100])
plt.xlabel('Lambda',fontsize=24)
plt.ylabel('',fontsize=24)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)

plt.savefig(PATH_RESULTS+'/boost.pdf',bbox_inches='tight')
plt.close()


# PREDICTED LAMBDA - HETEROGENEITY
print('running predicted lambda')

constraint_violations_lambda_predict = []
utility_lambda_predict = []
time_lambda_predict = []

l_models = []
for i in range(n_constraints):
    l_models.append(KNeighborsRegressor(n_neighbors=10, weights='distance').fit(
        X_all.iloc[users_train], lambda_all.iloc[users_train,i]))

for u in tqdm(users_test):

    U, A_list, b, X = extract_data(u, top_k=TOP_K, sample_size=SAMPLE_SIZE)

    start = time.time()
    lmbd = np.hstack([model.predict(np.expand_dims(X, axis=0)) for model in l_models])
    lmbd = np.maximum(lmbd,0)
    P, S = optimal_ranking(U, A_list, lmbd, method=METHOD,EPSILON=EPSILON)
    end = time.time()

    constraint_violations_u = [1*(np.trace(A_list[i].T.dot(P))<b[i]) for i in range(len(A_list))]
    constraint_violations_lambda_predict.append(constraint_violations_u)
    utility_lambda_predict.append(np.trace(U.T.dot(P)))

    time_lambda_predict.append(1000*(end - start))



# PREDICTED LAMBDA - AVG. LAMBDA
print('running avg. lambda')

constraint_violations_lambda_popul_avg = []
utility_lambda_popul_avg = []
time_lambda_popul_avg = []

np.random.seed(999)

lmbd_avg = lambda_all.iloc[users_train].mean(0)

for u in tqdm(users_test):

    U, A_list, b, X = extract_data(u, top_k=TOP_K, sample_size=SAMPLE_SIZE)

    start = time.time()
    P, S = optimal_ranking(U, A_list, lmbd_avg, method=METHOD,EPSILON=EPSILON)
    end = time.time()

    constraint_violations_u = [1*(np.trace(A_list[i].T.dot(P))<b[i]) for i in range(len(A_list))]
    constraint_violations_lambda_popul_avg.append(constraint_violations_u)
    utility_lambda_popul_avg.append(np.trace(U.T.dot(P)))

    time_lambda_popul_avg.append(1000*(end - start))



# encode numpy types to types that json understands
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


out = {}
out['EPSILON'] = EPSILON
out['No optimization'] = {'u':utility_no_optim, 
                    'cv':constraint_violations_no_optim,
                    'time':time_no_optim}
out['Optimal lambda'] = {'u':utility_lambda_optimal, 
                    'cv':constraint_violations_lambda_optimal,
                    'time':time_lambda_optimal}
out['KNeighbors lambda'] = {'u':utility_lambda_predict, 
                    'cv':constraint_violations_lambda_predict,
                    'time':time_lambda_predict}
out['Mean lambda'] = {'u':utility_lambda_popul_avg, 
                    'cv':constraint_violations_lambda_popul_avg,
                    'time':time_lambda_popul_avg}


with open(PATH_RESULTS+'/results_detailed.json', 'w') as json_file: 
    json.dump(out, json_file, cls=NpEncoder) 


# with open(PATH_RESULTS+'/results_detailed.json', "r") as read_file:
#     out = json.load(read_file)





