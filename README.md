# Tasting Movies

A project where we could find the perfect match of cine-fan and the next movie to watch.

Why? Well mainly because I love movies & love data and... who knows? maybe someone else like it also.


## Source

We'll use the [Kaggle challange for MovieLens 20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset)

- For the cleaning, feature engineering and the creation of the dataset look up [here](https://github.com/adfmb/tastingmovies/blob/main/0002_cleaning_and_featuring_engineering.ipynb)

- For the hyperparameteres tuning, the model fit, the feature importance and the final comments check out [here](https://github.com/adfmb/tastingmovies/blob/main/0002_cleaning_and_featuring_engineering.ipynb)

- For this first version it is important to say that it was chossen the use of:

  -  Hyperopt package o implement a bayesian optimization and in order to make the search smarter and efficient.
          
  -  XGBoost for model fit, since its way of build weak models provokes a very good learning with high performances and, last but not least, the ways ofr parallelize the work flow convert it in a very powerful and quick tool to build models
        
  -  Shap for the Featuring Importance
