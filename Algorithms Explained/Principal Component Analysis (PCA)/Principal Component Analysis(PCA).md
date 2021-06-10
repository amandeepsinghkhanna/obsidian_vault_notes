# Principal Component Analysis(PCA)

### Introduction:
Principal Component Analysis(PCA) is a dimensionality reduction algorithm. In the context of data science, each column in an excel sheet or a relational database table is considered to be a dimension. Dimensionality reduction is the process of reducing the number of columns/dimensions in the data such that there is an increase in the interpretability of the data while preserving maximum information. The reduced columns/dimensions obtained from the PCA algorithm are uncorrelated to each other. 

### Pre-requisites to understand the PCA algorithm:
1. <b>Standard deviation</b>:<br/>
Standard deviation is the measure of how spread out our data is. In other words we can say that standard deviation is the average distance from the mean of the data to a point. 

```Tex
StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}
```

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" title="\large StandardDeviation(\sigma)=\sqrt{\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}}" /></a>

2. <b>Variance</b>:<br/>
Variance is another measure of how spread out our data is. In fact it is just the square of the standard deviation. 

```Tex
Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}
```

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" title="\large Vaiance(\sigma^{2})=\frac{\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}{(n-1)}" /></a>

4. Covariance
5. Eigen values and eigen vectors






### WIP - Rough:

There are two methodologies for implementing the PCA algorithm:
1. By solving the eigen vector
2. By computing the Singular Value Decomposition(SVD)
