# How to win a Data Science Competitions: Learn from Top Kagglers
This document contains my notes from the course.

Expected time to complete: <b>54 Hours</b><br/>
Coursera link to the course: https://www.coursera.org/learn/competitive-data-science
<br/>
#### <u>Course Instructors</u>
- Dimitry Ulyanov
- Mikhail Torfimov
- Marios Michailidis
- Alexander Guschin
- Dimitry Altukhov

The course is offered by the HSE university from Russia.

#### <u>Weekwise Couse Syllabus:</u>

- <b>Week1</b>
    1. Intoduction to competitions & Recap
    2. Feature processing & extraction
- <b>Week2</b>
    1. Exploratory Data Analysis(EDA)
    2. Validation
    3. Data leaks
- <b>Week3</b>
    1. Metrics
    2. Mean-encodings
- <b>Week4</b>
    1. Advanced features
    2. Hyperparameter optimization
    3. Ensembles
- <b>Week5</b>
    1. Final project
    2. Winning solutions

### <u>Data Competition Platforms:</u>
- Kaggle
- DrivenData
- CrowdAnalityx
- CodaLab
- DataScienceChallenge.net
- Datascience.net

### <u>Difference between real world problems and contests:</u>

Real world problems required a complicated process to solve them with the following steps:

   1. Understanding the business problem
    2. Problem formulation
    3. Data collection
    4. Data preprocessing
    5. Modelling
    6. Methodology to evaluate the model in real life
    7. Methodology to deploy the model

It is a lot simpler process to solve a data science contest problem as it only follows the following steps:

   1. Data preprocessing
    2. Modelling

### <u>Important aspects to keep in mind while competing in data science competitions:</u>
1. It's all about the data and making things work and not about the algorithm itself.
2. Sometimes machine learning might not be required to win.
3. It doesn't matter if we use:
    - Heuristics
    - Manual data Analysis
    - Complex solutions
    - Advanced feature engineering
    - Doing huge calculations
4. The only things that matters is the target metric.

### <u>Families of Machine Learning Algorithms for Regression and Classification:</u>
- Linear
- Tree Based
- kNN
- Support Vector Machines
- Neural Networks

### <u>No Free Lunch Theorem:</u>
There is no machine learning algorithm that out-performs all others for all tasks.

### <u>Types of Features in data:</u>
1. Nominal
2. Ordinal
2. Numerical
3. Date-time
4. Co-ordinates

### Preprocessing Numerical Features:
There are models that depend and also don't depend on the scale of the numeric features.<br/>
kNN, Neural Networks, Linear Models and Support Vector Machines are effected by the scale of the numeric columns.<br/>
If we multiply a numeric feature with a large number then in a kNN model every minute difference will impact the prediction in a large way.<br/>
Linear models have trouble when all the features are scaled differently, first we want regularization to be applied to the coefficents of the linear model coefficents for features in equal amounts. But, regularizations impact turns out to be propotional to feature scale. Apart from this, gradient descent algorithm does not work well when features are not properly scaled.<br/>
Different feature scaling results in the different performance of the models, In this sense it is just another hyperparameter to be optimized.<br/>

1. <b>Min-Max Scaling:</b><br/>
All numeric values are transformed into values that lie between 0 and 1. 
<br/>

```python
sklearn.preprocessing.MinMaxScaler
```

X = (X - X.min())/(X.max() - X.min())

It is important to note that the underlying distribution of the column/variable will remain unchanged.

2. <b>Standard Scaling:</b><br/>
All numeric values are tranformed to have 0 mean and unit standard deviation.<br/>
```python
sklearn.preprocessing.StandardScaler
```
X = (X - X.mean())/X.std()

After the application of both Min-Max Scaling and Standard Scaling, feature impacts on the non-tree-based models will be similar.

3. <b>Outliers:</b><br/>
In linear models, the numeric columns are highly impacted/influenced by outliers. To protect the linear models from the impact of the outliers we clip the feature values between two chosen values i.e, a lower bound and a upper bound. We usually clip the lower outliers with the 1st percentile and the upper outliers with the 99th percentile.This process of clipping of outliers in numeric values in the financial data is called <b>winsorization</b>.

4. <b>Rank Transformation:</b><br/>
Another effective method for preprocessing numeric columns is rank transformation. Rank transformation sets spaces between proper assorted values to be equal. Rank transformation is a much better option for numerical columns with outliers as it moves outliers closer to the other objects.<br/>
Example of rank transformation:
```python
rank([-100, 0, 1e5]) == [0, 1, 2]
rank([1000, 1, 10]) == [2, 0, 1]
```
If we apply a rank to the source of an array, it will sort the array, changes values to their indices, establish mapping between the array values and their ranks and return the coresposing rank array. kNN and neural networks can benifit from this kind of transformation.
```python
scipy.stats.rankdata
```
5. <b>Log Transformations or Rasing the value to a power less than 1:</b><br/>
Another 2 techniques that particularly help the non-tree based models and especially neural networks.	

```python
# log transformation:
np.log(1+x)
# raising to the power<1:
np.sqrt(x+2/3)
```
	
Both of these transformation can be useful because they drive too big values closer to the features average value also they make the values near zero more distinguishable.

### <u>Preprocessing Categorical Features:</u>

1. <b>Label Encoding:</b><br/>
Ordinal features/columns are label encoded i.e, each category is replaced by a corresponding numerical value. This method works well with trees. Non-tree based models cannot utilize this type of pre-processed feature/variable appropriately. 
```python
sklearn.preprocessing.LabelEncoder
```
There are mainly 3 methods by which we perform label encoding:

  1. Label encoding can be applied in the alphebitical or sorted order.
  2. Label encoding can be applied in the order of apprearance.
  3. Each category can be replace by it corresponding frequency or proportion in the given column. One benifit of using frequency to encode the categorical variables is that when we have too many categories and if there are multiple categories with the same frequency then they would not be distinguishable. 


2. <b>Dummy Variables:(One-Hot Encoding)</b><br/>
Nominal features/columns are one-hot encoded i.e, each category has a seperate column created for it where 1 represents the occurance that category in a particular row and the 0 represents that the category has not occured in that particular row.
```python
pandas.get_dummies
sklearn.preprocessing.OneHotEncoder
```
Dummy variables suffer from a problem of sparsity of data.
						  
### Handling Missing Values:

Missing values could be not numbers, empty stings or outliers like -999.

Approaches to fill missing values:

1. -999, -1, etc - This approach creates a seperate category to represent missing values, but the performance of linear models might suffer.
2. mean, median, mode
3. Reconstruct value - Approximating the missing values by using the closes neighbours. This is very rarely possible.

<b>Handling categories in the test dataset that are missing from the training data:</b> <br/>In this context we apply frequency encoding on the categorical column.

It should be noted that Xgboost can handle missing or NaN values.

### Preprocessing datetime and coordinate features:

The features extracted from datetime features can be classified as:

1. Periodicity: Time moments in a given period i.e, Day in week, month, year, month in year, year, hours, minutes, seconds etc.
2. Time passed since a particular event: <br/>
	1. Row-independent moment. example: Time since 1st Jan 2020 till value in the row.
	2. Row-dependent moment. example: Number of days until next holiday.
3. Difference of two datetime features.

While dealing with coordinate data, if we have additional information about with infrastructural buildings we can add the distance to the nearest location of intrest. If we don't have such data, we can extract these relevant points from our training data. We could alternatively cluster the coordinate datapoints and treat the cluster centroids as the important data points to compute distance to a particular coordinate. Another feature that can be created is to aggregate the statistics and compute things like the number of resturants around that coordinate etc. 

### Feature extraction and preprocessing of textual information:

The steps involved in the preprocessing of textual information is as follows:
1. Lowercasing
2. Lemmatization
3. Stemming
4. Stopword Removal

The two main ways to extract features from the textual information are:
1. <b>Bag of Words:</b>
In a bag of words approach we create a new column for each unique for from our textual data, then we a metric like the count of occurances of each word and place these values in the appropriate columns.
```python
# Count Vectorization:
sklearn.feature_extraction.text.CountVectorizer
# Term Frequency - Inverse Document Frequency Vectorization:
# Assuming x is a pandas DataFrame containing word frequencies
tf = 1/x.sum(axis=1)[:, None]
x=x*tf
# Inverse Document Frequency:
idf = np.log(x.shape[0]/(x>0).sum(0))
x = x*idf
sklean.feature_extraction.text.TfidfVectorizer
```
Another method to extract features from textual information is to use Ngrams. In Ngrams, we not only add columns corresponding to the word, but also columns corresponding to inconsequent words. Careful text preprocessing can help the performance of the bag of words approach drastically.

2. <b>Word Embeddings</b>

 