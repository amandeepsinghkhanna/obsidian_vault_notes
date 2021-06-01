# How to win a Data Science Competition Learn from Top Kagglers

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

### <u>Families of Machine Learning Algorithms for Regression & Classification:</u>
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

### <u>Preprocessing Numerical Features:</u>
There are models that depend and also don't depend on the scale of the numeric features.<br/><br/>
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
5. <b>Log Transformations or Rasing the value to a power < 1:</b><br/>
Another 2 techniques that particularly help the non-tree based models and especially neural networks.
	
```python
# log transformation:
np.log(1+x)
# raising to the power<1:
np.sqrt(x+2/3)
```
						  
Both of these transformation can be useful because they drive too big values closer to the features average value also they make the values near zero more distinguishable.
						  
### <u>Handling Missing Values:</u>

Missing values could be not numbers, empty stings or outliers like -999.

Approaches to fill missing values:

1. -999, -1, etc - This approach creates a seperate category to represent missing values, but the performance of linear models might suffer.
2. mean, median, mode
3. Reconstruct value - Approximating the missing values by using the closes neighbours. This is very rarely possible.