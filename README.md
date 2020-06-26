# StarbucksCapstoneChallenge
[Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) 
[Capstone Project - Analyze Starbucks Capstone Challenge Dataset](https://medium.com/@bariqhibat/data-scientist-nanodegree-starbucks-capstone-challenge-2a313a5d4084)  

## Project Overview
Customer satisfaction drives business success and data analytics provides insight into what customers think. For example, the phrase "[360-degree customer view](https://searchsalesforce.techtarget.com/definition/360-degree-customer-view)" refers to aggregating data describing a customer's purchases and customer service interactions.
  
The Starbucks [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Capstone challenge data set is a simulation of customer behavior on the Starbucks rewards mobile application. Periodically, Starbucks sends offers to users that may be an advertisement, discount, or buy one get on free (BOGO). An important characteristic regarding this dataset is that not all users receive the same offer.
  
This data set contains three files. The first file describes the characteristics of each offer, including its duration and the amount  a customer needs to spend to complete it (difficulty). The second file contains customer demographic data including their age, gender, income, and when they created an account on the Starbucks rewards mobile application. The third file describes customer purchases and when they received, viewed, and completed an offer. An offer is only successful when a customer both views an offer and meets or exceeds its difficulty within the offer's duration.
  
## Problem Statement / Metrics 
The problem that I'm trying to solve here is classifying if customer will complete his offer or not, hence it is a classification problem. Before building the models, we will do some data cleaning. Because we are given three separate set of data, I'm trying to combine them into one final dataset. After combining the three of them, there are some problems with the datatypes. A column that meant to be a datetime data type, is in object type. There are also a few columns that needs to be made [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)), since it is a [categorical](https://en.wikipedia.org/wiki/Categorical_variable) column. After finish cleaning the data, we enter the second part of data science part, which is [data exploration](https://en.wikipedia.org/wiki/Data_exploration#:~:text=Data%20exploration%20is%20an%20approach,through%20traditional%20data%20management%20systems.). We're trying to explore the data by visualizing a few columns by making histograms, and barplots. Some plots are in [gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution), some are not. But definitely, there are some trends in the data. After finish exploring the data, we enter the machine learning part of the project.

All of the columns contribute to whether the customer will complete his offer or not. Additionally, the number of completed offer and incompleted offer is similar. This makes us not worry about having the classifier to have a high [bias](https://becominghuman.ai/machine-learning-bias-vs-variance-641f924e6c57) towards some class, because there are a small chance of one of them being an outlier. We're using [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) from [sklearn library](https://scikit-learn.org/stable/index.html) to assess the metrics of the machine learning model, which gives [precision, recall, and also f1-score](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/#:~:text=Precision%20%2D%20Precision%20is%20the%20ratio,the%20total%20predicted%20positive%20observations.&text=F1%20score%20%2D%20F1%20Score%20is,and%20false%20negatives%20into%20account.). We will compare each machine learning model, and lastly we will try using randomized search cross-validation with random forest classifier as classifier model. In conclusion, the steps that we're going to do this part is,
1. Clean the data
2. Explore the data
3. Split the data to train and test set
4. Train the model
5. Evaluation
6. Conclusion

## Results Summary
- Model ranking based on **f1-score**
    1. XGBoostClassifier (0.80)
    2. GradientBoostClassifier (0.79)
    3. RandomForestClassifier (0.78)
    4. LogisticRegression (0.75)
    5. GaussianNB (0.70)
    6. MultinomialNB (0.60)
- Model ranking based on **precision**
    1. XGBoostClassifier (0.80)
    2. GradientBoostClassifier (0.79)
    3. RandomForestClassifier (0.78)
    4. GaussianNB (0.77)
    5. LogisticRegression (0.75)
    6. MultinomialNB (0.60)
- Model ranking based on **recall**
    1. XGBoostClassifier (0.80)
    2. GradientBoostClassifier (0.79)
    3. RandomForestClassifier (0.78)
    4. LogisticRegression (0.75)
    5. GaussianNB (0.71)
    6. MultinomialNB (0.60)
- **This suggest that XGBoostClassifier performs the best**

At the end of the project, we also train a randomized search cv with RandomForestClassifier as its classifier. From there, we could see the [feature importances](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e) of the data. [Feature importances](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e) refers to a numerical value that describes a feature's contribution to building a model that maximizes its evaluation metric. My analysis of the Starbucks Capstone Challenge customer offer effectiveness suggests that the top five features based on their importance are:  
  
    1. Offer Reward 
    2. Offer Difficulty (how much money a customer must spend to complete an offer)  
    3. Offer Duration 
    4. Informational-type Offer  
    5. The year the customer joined as a customer

## Files  
- Starbucks_Capstone_notebook.ipynb  
  - [Jupyter notebook](https://jupyter.org/) that performs three tasks:  
    - Combines offer portfolio, customer demographic, and customer transaction data  
    - Generates training customer demographic data visualizations and computes summary statistics  
    - Generates machine learning models  
- data
  - portfolio.json
  - profile.json
  - transcript.json
- Starbucks_Capstone_notebook.html 
- README.md  
	
## Python Libraries Used
- [Python Data Analysis Library](https://pandas.pydata.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Scipy](https://www.scipy.org/)
- [Numpy](http://www.numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [seaborn: Statistical Data Visualization](https://seaborn.pydata.org/)  
- [re: Regular expression operations](https://docs.python.org/3/library/re.html)  
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)   
  
## References
- [Feature Importances](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e)
- [Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/#:~:text=Precision%20%2D%20Precision%20is%20the%20ratio,the%20total%20predicted%20positive%20observations.&text=F1%20score%20%2D%20F1%20Score%20is,and%20false%20negatives%20into%20account.)
- [360-degree customer view](https://searchcustomerexperience.techtarget.com/definition/360-degree-customer-view)
- [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- [Dummy Variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics))
- [Categorical Variables](https://en.wikipedia.org/wiki/Categorical_variable)
- [Data Exploration](https://en.wikipedia.org/wiki/Data_exploration#:~:text=Data%20exploration%20is%20an%20approach,through%20traditional%20data%20management%20systems.)
- [Classification Report](https://en.wikipedia.org/wiki/Data_exploration#:~:text=Data%20exploration%20is%20an%20approach,through%20traditional%20data%20management%20systems.)
- [Machine Learning: Bias VS. Variance](https://becominghuman.ai/machine-learning-bias-vs-variance-641f924e6c57)
- [Gaussian/Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)