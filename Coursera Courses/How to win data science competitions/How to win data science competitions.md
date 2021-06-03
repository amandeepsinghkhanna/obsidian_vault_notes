# How to win a Data Science Competitions: Learn from Top Kagglers
This document contains my notes from the course.

Expected time to complete: <b>54 Hours</b><br/>
Coursera link to the course: https://www.coursera.org/learn/competitive-data-science
<br/>
#### Course instructors
- Dimitry Ulyanov
- Mikhail Torfimov
- Marios Michailidis
- Alexander Guschin
- Dimitry Altukhov

The course is offered by the HSE university from Russia.

#### Weekwise course syllabus:

- <b>Week1</b>
    1. Introduction to competitions & Recap
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

### Data competition platforms:
- Kaggle
- DrivenData
- CrowdAnalityx
- CodaLab
- DataScienceChallenge.net
- Datascience.net

### Difference between real world problems and contests:

Real world problems required a complicated process to solve them with the following steps:

   1. Understanding the business problem
   2. Problem formulation
   3. Data collection
   4. Data preprocessing
   5. Modeling
   6. Methodology to evaluate the model in real life
   7. Methodology to deploy the model

It is a lot simpler process to solve a data science contest problem as it only follows the following steps:

   1. Data preprocessing
   2. Modeling

###  Important aspects to keep in mind while competing in data science competitions:
1. It's all about the data and making things work and not about the algorithm itself.
2. Sometimes machine learning might not be required to win.
3. It doesn't matter if we use:
    - Heuristics
    - Manual data Analysis
    - Complex solutions
    - Advanced feature engineering
    - Doing huge calculations
4. The only things that matters is the target metric.

### Families of machine learning algorithms for regression and classification:
- Linear
- Tree Based
- kNN
- Support Vector Machines
- Neural Networks

### No Free Lunch Theorem:
There is no machine learning algorithm that out-performs all others for all tasks.

### Types of features in data:
1. Nominal
2. Ordinal
2. Numerical
3. Date-time
4. Co-ordinates

### Preprocessing numerical features:
There are models that depend and also don't depend on the scale of the numeric features.<br/>
kNN, Neural Networks, Linear Models and Support Vector Machines are effected by the scale of the numeric columns.<br/>
If we multiply a numeric feature with a large number then in a kNN model every minute difference will impact the prediction in a large way.<br/>
Linear models have trouble when all the features are scaled differently, first we want regularization to be applied to the coefficients of the linear model coefficients for features in equal amounts. But, regularization's impact turns out to be proportional to feature scale. Apart from this, gradient descent algorithm does not work well when features are not properly scaled.<br/>
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
All numeric values are transformed to have 0 mean and unit standard deviation.<br/>
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
If we apply a rank to the source of an array, it will sort the array, changes values to their indices, establish mapping between the array values and their ranks and return the corresponding rank array. kNN and neural networks can benefit from this kind of transformation.
```python
scipy.stats.rankdata
```
5. <b>Log Transformations or raising the value to a power less than 1:</b><br/>
Another 2 techniques that particularly help the non-tree based models and especially neural networks.	

```python
# log transformation:
np.log(1+x)
# raising to the power<1:
np.sqrt(x+2/3)
```
	
Both of these transformation can be useful because they drive too big values closer to the features average value also they make the values near zero more distinguishable.

### Preprocessing categorical features:

1. <b>Label Encoding:</b><br/>
Ordinal features/columns are label encoded i.e, each category is replaced by a corresponding numerical value. This method works well with trees. Non-tree based models cannot utilize this type of preprocessed feature/variable appropriately. 
```python
sklearn.preprocessing.LabelEncoder
```
There are mainly 3 methods by which we perform label encoding:

  1. Label encoding can be applied in the alphabetical or sorted order.
  2. Label encoding can be applied in the order of appearance.
  3. Each category can be replace by it corresponding frequency or proportion in the given column. One benefit of using frequency to encode the categorical variables is that when we have too many categories and if there are multiple categories with the same frequency then they would not be distinguishable. 


2. <b>Dummy Variables:(One-Hot Encoding)</b><br/>
Nominal features/columns are one-hot encoded i.e, each category has a separate column created for it where 1 represents the occurrence that category in a particular row and the 0 represents that the category has not occurred in that particular row.
```python
pandas.get_dummies
sklearn.preprocessing.OneHotEncoder
```
Dummy variables suffer from a problem of sparsity of data.
						  
### Handling missing values:

Missing values could be not numbers, empty stings or outliers like -999.

Approaches to fill missing values:

1. -999, -1, etc - This approach creates a separate category to represent missing values, but the performance of linear models might suffer.
2. mean, median, mode
3. Reconstruct value - Approximating the missing values by using the closes neighbors. This is very rarely possible.

<b>Handling categories in the test dataset that are missing from the training data:</b> <br/>In this context we apply frequency encoding on the categorical column.

It should be noted that Xgboost can handle missing or NaN values.

### Preprocessing datetime and coordinate features:

The features extracted from datetime features can be classified as:

1. Periodicity: Time moments in a given period i.e, Day in week, month, year, month in year, year, hours, minutes, seconds etc.
2. Time passed since a particular event: <br/>
	1. Row-independent moment. example: Time since 1st Jan 2020 till value in the row.
	2. Row-dependent moment. example: Number of days until next holiday.
3. Difference of two datetime features.

While dealing with coordinate data, if we have additional information about with infrastructural buildings we can add the distance to the nearest location of interest. If we don't have such data, we can extract these relevant points from our training data. We could alternatively cluster the coordinate datapoints and treat the cluster centroids as the important data points to compute distance to a particular coordinate. Another feature that can be created is to aggregate the statistics and compute things like the number of restaurants around that coordinate etc. 

### Feature extraction and preprocessing of textual information:

The steps involved in the preprocessing of textual information is as follows:
1. Lowercasing
2. Lemmatization
3. Stemming
4. Stopword Removal

The two main ways to extract features from the textual information are:
1. <b>Bag of Words:</b>
In a bag of words approach we create a new column for each unique for from our textual data, then we a metric like the count of occurrences of each word and place these values in the appropriate columns.
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

Word embeddings are the numeric vectors representing the textual information that are generated using a neural network. The general intuition is that words that have similar meanings will often have similar numeric vector representations.

 A few examples for word embeddings that are used for words are:
 Word2Vec, Glove, FastText, etc
 
 A few examples of word embeddings that are used on sentences are:
 Doc2vec, etc 
 
 Word embeddings overcome the problem of sparsity from the bag of words approach.
 
 
 ### Feature generation and preprocesing images:
Convolutional neural networks (CNN's) can be used to receive compressed representation of an image. When we calculate the CNN's output for an image, besides getting the output on the last layer we also have outputs from the inner layers. These outputs from the inner layers are known as <b>Descriptors</b>.

The descriptors from outer layers are better to solve tasks similar to the ones that the neural network was trained on and the descriptors from the early layers have more of the task independent information and can be used for tasks that the neural network was not trained on.

To improve the performance of a neural network that is pre-trained we can tune the hyperparameters and this process is known as <b>Finetuning</b>. The process of finetuning the neural network is also known as <b>Transfer Learning</b>.

For smaller datasets, finetuning is much better than building a neural network from scratch. 

Sometimes image augmentation may be used to generate more images to train the neural network.

### Exploratory Data Analysis (EDA)
Exploratory data analysis is the process of looking into the data, understanding it and getting comfortable with it. When we look into the data, we get a better understanding of the data which leads to a better intuition about the data and then we can generate a hypothesis about the data and find insights. One of the major EDA tools is data visualization. When we visualize the data, we can immediately see patterns. 

In data science competitions we can use EDA to identify data leaks in our data like columns, or combinations of columns that act as a proxy of the target variable. 

Few things that make EDA better are:
1. Domain knowledge
2. Checking if the data is intuitive
3. Understanding how the data was generated

In the context of the data science competitions, it is essential to check if the distributions of all the features in the training dataset are similar to that of the testing dataset. 

#### EDA on anonymized data:
While dealing with anonymized data, we can try to guess the meaning and type of each feature individually. 

Subsequently, we could also find relations between pairs of features or find feature groups. 

A few helpful pandas functions to help facilitate better EDA are:
```python
df.dtypes # Type of each column in a pandas DataFrame.
df.info() # Datatype of each column with it's row count.
df.value_counts() # Unique values in a column with its counts.
x.isnull() # Checking for missing values in a column in a pandas DataFrame.
df.describe(include="all") # Generating summary statistics for all columns in the pandas DataFrame.
```

One of the key components of EDA is <b>Data Visualization</b>.

#### Data visualization:
Data visualizations can be broadly classified into 2 types:

1. Visualizations meant to explore individual features 
	1. Histograms
	2. Row index vs feature value as a scatter plot
	```python
	plt.hist(x)
	plt.plot(x, ".")
	```
2. Visualizations meant to explore feature relationships
	1.	Scatter Plots
	2.	Correlation Matrices

### Data cleaning
In our process to clean data we check for the following things:
1. Constant Features/Columns
2. Duplicated Features/Columns
3. Duplicated rows
4. If the dataset is shuffled

While checking for duplicate columns among categorical columns, we should look out for categories that have the different category labels but will be same as another column within the table. 
```python
# Python code to check for features that are similar but have different labels:
for feature in categorical_features:
	df[feature+"_encoded"] = df[feature].factorize()

df.T.drop_duplicates()
```

### Validation and overfitting:
We use validation to check the quality of the model that we have. Usually we split our data into two parts:

1. Training part
2. Validation part

We train the model on the training part and check it's quality on the validation part. Besides this our model will see the testing data in the future once we are satisfied by the performance of the model on the validation dataset. 

<b>Underfitting</b>

Under fitting is when our model performs badly on the training, validation and the test set. 

<b>Overfitting</b>

Overfitting is when our model performs very well on training data but performance drop drastically on the validation and test datasets. 

In order to build the best model we need to avoid both underfitting as well as overfitting. 

#### Validation strategies:

There are mainly three different validation strategies:

1. <b>Holdout</b> - There is one split in the data and one part is considered as training dataset and the other is considered as validation dataset such that there is no overlap within the training and validation.
2. <b>K-fold</b> - We split our data into k different parts and iterate through them using every part as validation set only once. It can be said that K-fold validation is k times hold out validation.
3. <b>Leave one out</b> - Leave one out validation is a special case of k-fold validation when k is equal to the number of samples in our data. We will iterate over each sample leaving k-1 objects as training and 1 object as testing subset.
4. <b>Moving window validation</b> - In case the dataset we are anlaysing is a time-series we use a moving window validation, where each period of time is used as a validation set.

The main difference between the three strategies is the number of splits that are made on the data.

When we are trying to randomly split the data which has less number of rows, a random split of the data can fail and we may have unequal distribution of target classes within each sample. In this case we will be splitting the data using stratification which will ensure that the distribution of the target categories is maintained across all the folds. 

#### User of stratification in sampling the data:
- Small datasets
- Unbalanced datasets
- Multiclass classification datasets with large number of categories

The train - validation split of the data should be set up in such a way that it mimics the train - test split. 

#### Splitting strategies in data science competitions:
Few of the most common splitting strategies used by organizers in data science competitions are:
1. Random, rowwise
2. Timewise
3. By id
4. Combined/Composite row id

#### Problems during validation of models:

The problems that we encounter while improving the performance of the model on the validation datasets can be broadly classified into 2  types:
1. <b>Problems during the validation sage</b> - To identify the problems in our training - validation split we will see a huge difference in errors/scores of the model for training dataset and validation dataset. We also see a huge difference in optimal parameters for the model between the training and the validation datasets.
2. <b>Problems during the submission stage</b> - To identify the problems in our validation split and the submission split we will see a huge difference between the errors/scores of the model on the validation dataset and the leader board.

### Metrics to evaluate machine learning models:

In the context of a data science contest, it is important to know the metric used for evaluation so that we can work on optimize the model to get the best results for the specified metric.

#### Regression metrics:

Few of the common regression metrics are:
1. Mean Squared Error (MSE)

$$MSE = \frac{1}{N} \Sigma^{N}_{i=1}(yi - \hat{yi})^{2}$$

3. Root Mean Squared Error (RMSE)
4. R-squared
5. Adjusted R-squared
6. Mean Absolute Error (MAE)
7. Mean Absolute Percentage Error (MAPE)
8. (MSPE)
9. (MSLE)

#### Classification metrics:

1. Accuracy
2. Log loss
3. Area under the curve
4. Cohen's Kappa
