# importing the dependencies


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Processing


# Load the Data from CSV File to Pandas DataFrame
titanic_data = pd.read_csv("/Users/DELL/PycharmProjects/pythonProject10/train.csv")
# Printing the First Five Rows of the Dataframe
titanic_data.head()
# number of rows and columns
titanic_data.shape
# getting some information about the data
titanic_data.info()
# checking the number of missing values in each column
titanic_data.isnull().sum()


# Handling the missing values


# drop the cabin column from he dataframe
titanic_data = titanic_data.drop(columns="Cabin", axis=1)                # 0 for row, 1 for column
# assign mean age in missing age places
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)     # fillna for filling missing values
# inplace to save in original dataframe
# finding the mode value for embarked column
print(titanic_data["Embarked"].mode())
print(titanic_data["Embarked"].mode()[0])
# replacing the missing values in Embarked column with mode value
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)
titanic_data.isnull().sum()
# getting some statistical measures about the data
titanic_data.describe()
# finding the number of people who survived and didn't survive
titanic_data["Survived"].value_counts()                                   # 0 for not survived, 1 for survived


# Data Visualization


sns.set()
# Making a count plot for survived column
sns.countplot(x="Survived", data=titanic_data)
# Making a count plot for sex column
sns.countplot(x="Sex", data=titanic_data)
titanic_data["Sex"].value_counts()
sns.countplot(x="Sex", hue="Survived", data=titanic_data)
plt.show()
# Making a count plot for Pclass column
sns.countplot(x="Pclass", data=titanic_data)
sns.countplot(x="Pclass", hue="Survived", data=titanic_data)
plt.show()

# Encoding the Categorical Columns


# Replacing male values with 0, female values with 1
titanic_data["Sex"].value_counts()
titanic_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)

#
titanic_data["Embarked"].value_counts()
