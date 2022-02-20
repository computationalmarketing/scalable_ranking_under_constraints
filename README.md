# Scalable Ranking under Constraints

The repository contains code to reproduce results in the [working paper](https://arxiv.org/abs/2202.07088) `Yegor Tkachenko, Wassim Dhaouadi, and Kamel Jedidi. Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction (2022).`


Here is the bibtex citation format:
```
@article{scalableranking2022,
author  = {Tkachenko, Yegor and Dhaouadi, Wassim and Jedidi, Kamel},
title   = {{Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction}},
journal = {arXiv preprint arXiv:2202.07088},
year    = {2022}}
```

The paper describes a methodology for real-time optimal [re-ranking](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) of content candidates based on multiple objectives -- as the final stage in a recommender system.

`./code/environment.yml` is a conda environment file you can use to set up the programming environment.

You will need to download [GroupLens 25M](https://grouplens.org/datasets/movielens/25m/) and [Yow News](https://users.soe.ucsc.edu/~yiz/papers/data/YOWStudy/) data sets such that you get the following directory structure.

- data
	- yow-dataset
		- `yow_userstudy_raw.xls`
	- ml-25m
		- `ratings.csv`
		- `movies.csv`
		- `genome-scores.csv`
		- ...
- code

You will then run the python code files from the code directory in the following order to generate the data, run experiments, and visualize the results (`run.sh` file is provided for convenience):

- `prepare_data_mf_movies.py`
- `prepare_data_mf_news.py`
- `ech_movies_50.py`
- `ech_movies_500.py`
- `ech_movies_1000.py`
- `ech_news_50.py`
- `ech_news_500.py`
- `ech_news_1000.py`
- `summarize_results.py`

**Note:** 

- We use [CBC solver](https://projects.coin-or.org/Cbc), via cvxpy interface, as can be seen in `core_functions_unbalanced.py`. The solver needs to be installed separately.
- Variable EPSILON in code refers to <img src="https://latex.codecogs.com/svg.latex?1+\varepsilon"/> quantity in the paper.


Copyright (C) 2022 [Yegor Tkachenko](https://yegortkachenko.com), [Wassim Dhaouadi](https://www.gsb.stanford.edu/programs/phd/academic-experience/students/wassim-dhaouadi), [Kamel Jedidi](https://www8.gsb.columbia.edu/cbs-directory/detail/kj7)