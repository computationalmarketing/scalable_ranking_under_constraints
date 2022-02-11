
# Copyright (C) 2022 Yegor Tkachenko, Wassim Dhaouadi, Kamel Jedidi
# Code -- Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction
# https://github.com/computationalmarketing/scalable_ranking_under_constraints/

# Code preparing News Data
# https://users.soe.ucsc.edu/~yiz/papers/data/YOWStudy/

import numpy as np
import pandas as pd
import torch

from scipy import linalg
from scipy.sparse import csr_matrix, find
from pandas.api.types import CategoricalDtype

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import multiprocessing

from functools import partial

from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import json

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA = torch.cuda.is_available()

np.random.seed(999)


path = '../data/yow-dataset'

data = pd.read_excel(path+'/yow_userstudy_raw.xls')
data = data[['user_id', 'DOC_ID', 'relevant', 'classes']]
data = data[~data['relevant'].isna()]
data = data[data['relevant']!=-1]

data['relevant'] = data['relevant'].astype('int64')-1


n_ratings = data['relevant'].unique().shape[0]
n_items = data['DOC_ID'].unique().shape[0] #5827
n_users = data['user_id'].unique().shape[0] #24

mid_to_ind = dict(zip(sorted(data['DOC_ID'].unique()), range(n_items)))
ind_to_mid = dict(zip(range(n_items), sorted(data['DOC_ID'].unique())))


uids = data['user_id'].unique()
np.random.shuffle(uids)
uid_to_ind = dict(zip(uids, range(n_users)))
ind_to_uid = dict(zip(range(n_users), uids))

data.DOC_ID = data.DOC_ID.map(mid_to_ind)
data.user_id = data.user_id.map(uid_to_ind)


# embeddings
class EmbeddingNet(nn.Module):

    def __init__(self, n_users, n_items, n_ratings, n_dim=20):     
        super().__init__()
        
        # syllable emebeddings
        self.u_emb = nn.Embedding(n_users, n_dim)
        self.u_intercept = nn.Embedding(n_users, n_ratings)

        self.i_emb = nn.Embedding(n_items, n_dim)
        self.i_intercept = nn.Embedding(n_items, n_ratings)
        
        # utility function
        self.transform = nn.Sequential(
            nn.Linear(2*n_dim, n_ratings*3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_ratings*3, n_ratings))
                
    def forward(self, users, items):
        u_emb = self.u_emb(users)
        i_emb = self.i_emb(items)
        x = torch.cat([u_emb, i_emb], dim=1)
        out = self.transform(x) + self.u_intercept(users) + self.i_intercept(items)
        return out

# cross-entropy loss function
def loss_f(model, var):
    
    data, target = var
    data, target = data.long(), target.long()
    if CUDA:
        data, target = data.cuda(), target.cuda()
        
    users = data[:,0]
    movies = data[:,1]
    
    r = target
    pr_r = model(users, movies)

    out = F.cross_entropy(pr_r, r) 
    return out


#finetune and save neural net model
def finetune_and_save(model, loader_train, loader_test, savedir, model_name='finetuned_model'):

    if CUDA:
        model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.0001)

    hist = {}
    hist['train_loss'] = []
    hist['val_loss'] = []

    # train and evaluate
    for epoch in tqdm(range(NUM_EPOCHS)):
        
        train_loss = run_epoch(model, loss_f, optimizer, loader_train, update_model = True) # training
        eval_loss = run_epoch(model, loss_f, optimizer, loader_test, update_model = False) # evaluation

        print('epoch: {} \ttrain loss: {:.6f} \tvalidation loss: {:.6f}'.format(epoch, train_loss, eval_loss))

        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(eval_loss)

        with open(savedir+'/' + model_name + '_eval_record.json', 'w') as fjson:
            json.dump(hist, fjson)

    # saving model
    torch.save(model, savedir+"/"+model_name)
    return


# function that performa training (or evaluation) over an epoch (full pass through a data set)
def run_epoch(model, loss_f, optimizer, loader, update_model = False):

    if update_model:
        model.train()
    else:
        model.eval()

    loss_hist = []

    for batch_i, var in tqdm(enumerate(loader)):

        loss = loss_f(model, var)

        if update_model:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.data.item())

    return np.mean(loss_hist).item()


BATCH_SIZE = 200
NUM_EPOCHS = 5

np.random.seed(999)
torch.manual_seed(999)

X = data[['user_id','DOC_ID']].values
Y = data['relevant'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.from_numpy(X_train).long()
X_test = torch.from_numpy(X_test).long()
Y_train = torch.from_numpy(Y_train).long()
Y_test = torch.from_numpy(Y_test).long()

train = torch.utils.data.TensorDataset(X_train, Y_train)
test = torch.utils.data.TensorDataset(X_test, Y_test)
loader_train = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
loader_test = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

model = EmbeddingNet(n_users=n_users, n_items=n_items, n_ratings=n_ratings, n_dim=20)

finetune_and_save(model, loader_train, loader_test, savedir=path)

# load model
model = torch.load(path+'/finetuned_model')
model.eval()


# torch.argmax(x, dim=1)
# F.softmax(x,dim=1).detach().numpy().dot(np.array([1,2,3,4,5]))
# user_embeddings = list(model.intercept.parameters())[0].data.numpy()

x = model(X_test[:,0].cuda(),X_test[:,1].cuda()).cpu()
prd = torch.argmax(x,dim=1).detach().numpy()
tr = Y_test.detach().numpy()
print("Accuracy: ", np.mean(prd==tr))
# F.softmax(x,dim=1).detach().numpy().dot(np.array([1,2,3,4,5]))

# user factors
user_factors = np.concatenate((list(model.u_emb.parameters())[0].cpu().data.numpy(),list(model.u_intercept.parameters())[0].cpu().data.numpy()),1)

# cluster users
np.random.seed(999)
pca = PCA(n_components=5)
X_embedded = pca.fit_transform(user_factors)
pca.explained_variance_ratio_

clust = KMeans(n_clusters=3, random_state=0).fit(X_embedded)
labels = clust.labels_

plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
plt.savefig(path+'/clusters_users.pdf')
plt.close()


# get relevance
np.random.seed(999)

model.cpu()
model.eval()

u_indices = np.repeat(range(n_users),n_items)
i_indices = np.tile(range(n_items),n_users)

r = []
for u in tqdm(range(n_users)):
    ui = np.repeat(u,n_items)
    x = model(torch.from_numpy(ui).long(),torch.from_numpy(np.array(range(n_items))).long())
    r.append(F.softmax(x,dim=1).detach().numpy().dot(np.array([1,2,3,4,5])))
    #r.append(torch.argmax(x,dim=1).detach().numpy()+1)

r = np.concatenate(r,0)

generated_data = pd.DataFrame(np.stack([u_indices,i_indices,r.flatten()]).T)
generated_data.columns = ['user_id', 'DOC_ID', 'relevant']



# ADDING COVARIATES

## Look for the "science-technology" topics
idx_science_tech = data['classes'].str.contains('science|tech|medicine|discoveries|amazon|google|yahoo|microsoft|space|stem|studies',na=False,case=False)
articles_science_tech = list(set(data[idx_science_tech]['DOC_ID']))

## Look for the "Entertainment" topics
idx_entr=data['classes'].str.contains('art|book|fashion|culture|show|entertain|broadway|television|music|movies|theatre',na=False,case=False)
articles_entr=list(set(data.loc[idx_entr]['DOC_ID']))

## Look for the "World" topics
idx_world=data['classes'].str.contains('Iraq|World|qaeda|war|interna|china|military|terror',na=False,case=False)
articles_world=list(set(data.loc[idx_world]['DOC_ID']))

## Look for the "Health" topics
idx_health=data['classes'].str.contains('health|medicine|food|drugs',na=False,case=False)
articles_health=list(set(data.loc[idx_health]['DOC_ID']))

## Look for the "Business" topics
idx_business=data['classes'].str.contains('business|finan|econom',na=False,case=False)
articles_business=list(set(data.loc[idx_business]['DOC_ID']))

## Look for the "politics" topics
idx_politics=data['classes'].str.contains('election|politic|usa|president',na=False,case=False)
articles_politics=list(set(data.loc[idx_politics]['DOC_ID']))

## Look for the "sport" topics
idx_sport=data['classes'].str.contains('sport|ball|hockey|golf',na=False,case=False)
articles_sport=list(set(data.loc[idx_sport]['DOC_ID']))

## Look for the "environment" topics
idx_env=data['classes'].str.contains('envir|natur|energ|warming|pollut',na=False,case=False)
articles_env=list(set(data.loc[idx_env]['DOC_ID']))

columns = ['Science_Technology', 'Health', 'Business', 'Entertainment', 'World', 'Politics', 'Sport', 'Environment']
columns_articles_var=[articles_science_tech,articles_health,articles_business,articles_entr,articles_world,articles_politics,articles_sport,articles_env]

for i,v in enumerate(columns_articles_var):
    generated_data[columns[i]]=1*(generated_data['DOC_ID'].isin(v))


generated_data = pd.concat([generated_data,
    pd.DataFrame(user_factors[generated_data['user_id'].astype('int64')],columns=['e'+str(i) for i in range(25)])],1)

generated_data['userClust'] = labels[generated_data['user_id'].astype('int64')]

generated_data.to_csv(path+'/generated_data.csv', index=False)



