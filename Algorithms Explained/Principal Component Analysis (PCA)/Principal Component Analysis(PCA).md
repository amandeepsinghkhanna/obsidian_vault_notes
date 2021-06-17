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

# ROUGH SECTION: ALL THE WIP NOTES:

There are two methodologies for implementing the PCA algorithm:
1. By solving the eigen vector
2. By computing the Singular Value Decomposition(SVD)
