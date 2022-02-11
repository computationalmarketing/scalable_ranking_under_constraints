#!/bin/sh

# export conda environment file
# conda env export > environment.yml
# conda env create -f environment.yml

# prepare training data
python prepare_data_mf_news.py
python prepare_data_mf_movies.py

# run optimization experiments
python ech_news_50.py
python ech_news_500.py
python ech_news_1000.py

python ech_movies_50.py
python ech_movies_500.py
python ech_movies_1000.py

# summarize results
# python summarize_results.py