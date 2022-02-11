
# Copyright (C) 2022 Yegor Tkachenko, Wassim Dhaouadi, Kamel Jedidi
# Code -- Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction
# https://github.com/computationalmarketing/scalable_ranking_under_constraints/

# Code analyzing the results



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

import json

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_theme(style="whitegrid")
import os

if not os.path.exists('../results'):
    os.makedirs('../results')

PATH_RESULTS = '../results/summary'
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

colors = ['#e6194B', '#3cb44b',  '#4363d8', '#911eb4', '#42d4f4', '#f032e6', '#ffe119', '#bfef45', '#fabed4', '#469990', '#f58231',  '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

shapes = ['o', '^', 's', 'd']

def out_stargazer_latex(ress, colnames, title):

    #summary_col([res],stars=True).as_latex()
    stargazer = Stargazer(ress)
    stargazer.custom_columns(colnames, [1 for i in colnames])
    stargazer.show_model_numbers(False)
    stargazer.dependent_variable_name("Reconstruction metric")
    #stargazer.custom_columns('Variable reconstruction')
    stargazer.significant_digits(2)
    stargazer.show_confidence_intervals(True)
    stargazer.title(title)
    #stargazer.rename_covariates({'Age': 'Oldness'})
    #stargazer.covariate_order(['BMI', 'Age', 'S1', 'Sex'])
    #stargazer.add_custom_notes(['First note', 'Second note'])
    print(stargazer.render_latex())

paths = ['../results/ml-25m-50/results_detailed.json',
    '../results/ml-25m-500/results_detailed.json',
    '../results/ml-25m-1000/results_detailed.json',
    '../results/yow-dataset-50/results_detailed.json',
    '../results/yow-dataset-500/results_detailed.json',
    '../results/yow-dataset-1000/results_detailed.json']


dt = []
for p in paths:
    nam = p.split('/')[2]
    with open(p, "r") as read_file:
        temp = json.load(read_file)

    del temp['EPSILON']

    for k in temp.keys():
        temp[k]['cv'] = [np.mean(i) for i in temp[k]['cv']]

    for k in temp.keys():
        d = pd.DataFrame.from_dict(temp[k])
        d['algorithm'] = k
        d['experiment'] = nam
        dt.append(d)

dt = pd.concat(dt)
dt['cc'] = 1.-dt['cv']

dt['time'] = np.log10(dt['time'])
dt['data'] = dt['experiment'].str.contains('yow')
dt.loc[dt['experiment'].str.contains('yow'),'data'] = 'yow'
dt.loc[~dt['experiment'].str.contains('yow'),'data'] = 'movielens'
dt['size'] = dt['experiment'].map(lambda x: int(float(x.split('-')[-1])))

# pd.pivot_table(dt, 
#     values='u', 
#     index='experiment', 
#     columns='algorithm', 
#     aggfunc='mean').reset_index()

# pd.pivot_table(dt, 
#     values='cc', 
#     index='experiment', 
#     columns='algorithm', 
#     aggfunc='mean').reset_index()


# MOVIELENS DATA COMPARISON

m = dt[dt['data']=='movielens'].groupby(['algorithm','size']).mean().reset_index()
se = (2*dt[dt['data']=='movielens'].groupby(['algorithm','size']).sem()).reset_index()

res = m.merge(se, how='inner', 
    on=['algorithm','size'],
    suffixes=('_m', '_2se'))

# color/shape assignment
al = sorted(res['algorithm'].unique().tolist())
alg_colors = dict(zip(al,colors[:len(al)]))
alg_shapes = dict(zip(al,shapes[:len(al)]))
sz = sorted(res['size'].unique().tolist())
sz_shapes = dict(zip(sz,shapes[:len(sz)]))

fig, ax = plt.subplots(figsize=(5, 2.5))

for i in range(len(al)):

    temp = res[res['algorithm']==al[i]]
    x = temp['time_m'].tolist()
    y = temp['cc_m'].tolist()
    x_err = temp['time_2se'].tolist()
    y_err = temp['cc_2se'].tolist()
    algs = res['algorithm'].tolist()
    s = temp['size'].tolist()

    for j in range(len(s)):
        ax.errorbar(x[j], y[j], xerr = x_err[j], 
                 yerr = y_err[j],
                 fmt=sz_shapes[s[j]], 
                 color=alg_colors[al[i]],
                 #label=al[i] + ' ' + str(s[j]),
                 elinewidth=1.0, ms=7.)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: r'${}^{}$'.format(10,int(x))))
plt.xlabel('Milliseconds',fontsize=12)
plt.ylabel('Constraint compliance probability',fontsize=12)
# ax.legend(loc='lower right')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=np.log10(50),c='black',ls='--') # 50 ms
plt.annotate('50 ms.', (np.log10(70), 0.85),fontsize=12,c='black')

#tikzplotlib.save(PATH_RESULTS+'/comparison.tex')
plt.savefig(PATH_RESULTS+'/comparison_movielens.pdf',bbox_inches='tight')
plt.close()



# LEGEND - COLORS
fig, ax = plt.subplots(figsize=(0.2, 1.5))
legend_elements = []
for i in range(len(al)):
    legend_elements.append(Line2D([0], [0], 
        marker="s", color=alg_colors[al[i]], 
        label=al[i], markersize=7., ls="none"))
ax.legend(handles=legend_elements, loc='center', title=r'$\textbf{Algorithm}$', title_fontsize=12, ncol=4)
plt.gca().set_axis_off()
plt.savefig(PATH_RESULTS+'/legend_colors.pdf',bbox_inches='tight')
plt.close()


# LEGEND - SHAPES
fig, ax = plt.subplots(figsize=(0.2, 1.5))
legend_elements = []
for i in range(len(sz)):
    legend_elements.append(Line2D([0], [0], 
        marker=sz_shapes[sz[i]], color='black', 
        label=sz[i], markersize=7., ls="none"))
ax.legend(handles=legend_elements, loc='center', title=r'$\textbf{Problem size}$', title_fontsize=12, ncol=3)
plt.gca().set_axis_off()
plt.savefig(PATH_RESULTS+'/legend_shapes.pdf',bbox_inches='tight')
plt.close()




# YOW News
m = dt[dt['data']=='yow'].groupby(['algorithm','size']).mean().reset_index()
se = (2*dt[dt['data']=='yow'].groupby(['algorithm','size']).sem()).reset_index()

res = m.merge(se, how='inner', 
    on=['algorithm','size'],
    suffixes=('_m', '_2se'))

# color/shape assignment
al = sorted(res['algorithm'].unique().tolist())
alg_colors = dict(zip(al,colors[:len(al)]))
alg_shapes = dict(zip(al,shapes[:len(al)]))
sz = sorted(res['size'].unique().tolist())
sz_shapes = dict(zip(sz,shapes[:len(sz)]))

fig, ax = plt.subplots(figsize=(5, 2.5))

for i in range(len(al)):

    temp = res[res['algorithm']==al[i]]
    x = temp['time_m'].tolist()
    y = temp['cc_m'].tolist()
    x_err = temp['time_2se'].tolist()
    y_err = temp['cc_2se'].tolist()
    algs = res['algorithm'].tolist()
    s = temp['size'].tolist()

    for j in range(len(s)):
        ax.errorbar(x[j], y[j], xerr = x_err[j], 
                 yerr = y_err[j],
                 fmt=sz_shapes[s[j]], 
                 color=alg_colors[al[i]],
                 elinewidth=1.0, ms=7.)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,y: r'${}^{}$'.format(10,int(x))))
plt.xlabel('Milliseconds',fontsize=12)
plt.ylabel('Constraint compliance probability',fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=np.log10(50),c='black',ls='--') # 50 ms
plt.annotate('50 ms.', (np.log10(70), 0.85),fontsize=12,c='black')

#tikzplotlib.save(PATH_RESULTS+'/comparison.tex')
plt.savefig(PATH_RESULTS+'/comparison_yow.pdf',bbox_inches='tight')
plt.close()






model_u = smf.ols(formula="u ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='movielens'])
model_u = model_u.fit(cov_type = 'HC3')

model_cc = smf.ols(formula="cc ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='movielens'])
model_cc = model_cc.fit(cov_type = 'HC3')

model_time = smf.ols(formula="time ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='movielens'])
model_time = model_time.fit(cov_type = 'HC3')

out_stargazer_latex([model_time,model_cc,model_u], ['Computing Time', 'Constraint Compliance', 'Utility'], # #, 
    title='Algorithm comparison (MovieLens)')


model_u = smf.ols(formula="u ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='yow'])
model_u = model_u.fit(cov_type = 'HC3')

model_cc = smf.ols(formula="cc ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='yow'])
model_cc = model_cc.fit(cov_type = 'HC3')

model_time = smf.ols(formula="time ~ C(size) + C(data) + C(algorithm, Treatment(reference='KNeighbors lambda'))", 
    data=dt[dt['data']=='yow'])
model_time = model_time.fit(cov_type = 'HC3')

out_stargazer_latex([model_time,model_cc,model_u], ['Computing Time', 'Constraint Compliance', 'Utility'], # #, 
    title='Algorithm comparison (YOW news)')



