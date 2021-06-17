# Principal Component Analysis(PCA)

### Introduction:

<div align="justify">
Principal Component Analysis(PCA) is a dimensionality reduction algorithm. In the context of data science, each column in an excel sheet or a relational database table is considered to be a dimension. Dimensionality reduction is the process of reducing the number of columns/dimensions in the data such that there is an increase in the interpretability of the data while preserving maximum information. The reduced columns/dimensions obtained from the PCA algorithm are uncorrelated to each other. 
</div>

### Pre-requisites for understanding the PCA algorithm:

Like all algorithms in data science, the PCA algorithm also requires us to know a few basic concepts in statistics and linear algebra. 

#### Statistical Concepts:

1. <b>Standard deviation</b>:<br/>
<div align="justify">Standard deviation is the measure of how spread out our data is. In other words we can say that standard deviation is the average distance from the mean of the data to a point.</div>

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" title="\large StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" /></a>

2. <b>Variance</b>:<br/>
<div align="justify">Variance is another measure of how spread out our data is. In fact it is just the square of the standard deviation. Variance is more sensitive to outliers.</div>

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" title="\large Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" /></a>

3. <b>Covariance</b>:<br/>
<div align="justify">Standard deviation and variances are always calculated with respect to only one dimension where as covariance is always calculated between any 2 given dimensions. It can be said that covariance is the variance between any 2 given columns. Covariance can be referred to as the degree of linear relationship between any two given dimensions.</div>

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;cov(x,y)&space;=&space;\frac{\sum_{i=1}^{n}(X_{i}&space;-&space;\overline{X})(Y_{i}&space;-&space;\overline{Y})}{(n-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;cov(x,y)&space;=&space;\frac{\sum_{i=1}^{n}(X_{i}&space;-&space;\overline{X})(Y_{i}&space;-&space;\overline{Y})}{(n-1)}" title="\large cov(x,y) = \frac{\sum_{i=1}^{n}(X_{i} - \overline{X})(Y_{i} - \overline{Y})}{(n-1)}" /></a>

#### Linear Algebra Concepts:

1. <b> Eigen Values & Eigen Vectors</b>:</br>


### The Algorithm:

### Using Eigen value and vectors:

<b>Step 1</b>: Start</br>
<b>Step 2</b>: Read the input dataset.</br>
<b>Step 3</b>: Compute the number of rows and columns in the input dataset.</br>
<b>Step 4</b>: Compute the minimum value between the number of rows and columns of the input dataset as max_possible_components.</br>
<b>Step 5</b>: Input the number of principal components to be extracted from the dataset as n_components.</br>
<b>Step 6</b>: If n_components is between 1 and max_possible_components proceed to <b>Step 7</b> else move to <b>Step 22</b>.</br>
<b>Step 7</b>: For each column/dimension in the input dataset compute the maximum, minimum and their average values.</br>
<b>Step 8</b>: For each column/dimension in the input dataset subtract each value with the column's maximum value and divide it by the difference of the column's respective maximum and minimum value. </br>
<b>Step 9</b>: For each column/dimension in the input dataset subtract each value with the column's average value. </br>
<b>Step 10</b>: For each combination of column/dimension in the input dataset compute its respective covariance and store the output as a covariance_matrix.</br>
<b>Step 11</b>: Compute eigen vector and values for the covariance_matrix.</br>
<b>Step 12</b>: Sort the the groups of eigen values & vectors in the decreasing order of their eigen values as feature_vectors.</br>
<b>Step 13</b>: Subset the feature_vectors for the first n_components number of elements as the subsetted_feature_vectors.</br>
<b>Step 14</b>: Compute the dot product of the input dataset with the subsetted_feature_vectors.</br>