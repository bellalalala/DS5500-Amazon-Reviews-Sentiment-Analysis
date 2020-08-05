# Amazon Reviews Sentiment Analysis
## Introduction
Our project used a large dataset(https://nijianmo.github.io/amazon/index.html), which contains over 20 million reviews from 24 departments to create a user interface, that could help amazon merchant to deal with large amount of reviews everyday.

## Requirements
```
dash==1.0.2
dash-daq==0.1.7
numpy>=1.16.2
pandas>=0.24.2
```

To install all of the required packages, run 
```
pip install -r requirements.txt
```

For installing fasttext module, check the official documentation(https://fasttext.cc/docs/en/support.html)

## Notebook
All code written in jupyter notebook is stored in `notebook/` folder. 

`preprocessing.ipynb` contains all code that processing the data before we plotting and training the model, the original data is downloaded from the link we give in the first line.

`basic EDA.ipynb` contains all code that used in first stage, EDA part, we explore every feature in our dataset, and come out with two problems that we are interested, can we define TRUE customer and how does number of reviews change among the time. 

`Model_Training.ipynb` contains all code that used to train the baseline and fasttext model. The training set and test set data is download from (https://www.kaggle.com/bittlingmayer/amazonreviews)


## Dashboard
All code and necessary file used for dashboard is stored in `dashboard/` file. In order to run the dashboard in your local computer, use following commands:
```
cd dashbard
python app.py
```
folder `dashboard_data/` contains the pre-trained model and a dictionary of weights that used to draw word cloud.

The user guide of dashboard can be found in `user_manual.md`.