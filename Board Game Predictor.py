#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To check whether the dataSet is in the same file or not.

import os
print(os.listdir())


# In[2]:


# Checking the version of Python we are using

import sys
print(sys.version)


# In[3]:


# Importing the essential python libraries

import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# To get the graphs inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the data.
games = pandas.read_csv("games.csv")
# Print the names of the columns in games.
print(games.columns)
print(games.shape)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
plt.show()


# In[4]:


#Checking the dataSet 
games.head(10)


# In[5]:


# To check the info of the dataSet
games.info()


# In[6]:


# To print the first row of all the games which has average rating equal to zero
print(games[games["average_rating"] == 0].iloc[0])


# In[7]:


# To print the first row of all the games which has average rating greater than zero
print(games[games["average_rating"] > 0].iloc[0])


# In[8]:


# Want to remove the rows which was without rating
games = games[games["users_rated"] > 0]

#Remove the rows with missing values
games = games.dropna(axis = 0)


# In[9]:


# Making a histogram with the average rating again 
plt.hist(games['average_rating'])


# In[10]:


#Using Seaborn to get the Correlation Matrix
corrmat = games.corr()
fig = plt.figure(figsize= (12,10))
sns.heatmap(corrmat, vmax = .8, square = True)


# In[11]:


# Get all the columns from the dataframe.
columns = games.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on.
target = "average_rating"


# # Linear Regression

# In[12]:


# Import a convenience function to split the sets.
from sklearn.model_selection import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# In[13]:


# Import the linear regression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test[target])


# # Random Forest

# In[14]:


# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])

