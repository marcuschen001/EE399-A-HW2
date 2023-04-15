# EE399-A-HW2

## Correlation
Marcus Chen, April 14, 2023

Correlation is a mutual relationship between two or more things; it is one of the core activities of unsupervised machine learning training processes. In this project, we will look at datasets containing pixel intensity values corresponding to face information and look at how correlation can be used to compare datasets. We will also explore correlation that is realized due to data reduction algorithms. 

### Introduction and Overview:
In the last project, we used one-dimensional data in order to look at the effects of optimization and model selection for curve fitting models, but what does a computer do with information that is more complex: such as the pixel intensity values of a face? Correlation is one of the main methods that we have in unsupervised machine learning to find general relationships between high-dimensional data and can be done directly or through single value decomposition.

In this project, faces from a Yale database are used to explore correlation. The dataset, X, contains over 2,414 face images with 1,024 pixel intensity values (corresponding to a 32 ✕ 32 image), but for the first experiment we will only be using the first 100 face images.

The 100 faces are turned into a correlation matrix using the Pearson Product-moment Correlation algorithm, which can then be used to find the two faces that share the most correlation and the two faces that share the least. 

Next, the same dataset is used but only for these images:
```
[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
```

For the next experiment, the dataset X is multiplied by its transpose so that we can use a simple correlation matrix to determine the eigenvalues and eigenvectors - or core characteristics - of the data, ranked by the eigenvalue. 

X is also sent through single value decomposition (SVD) in order to get the principal component directions, also ranked by size of the separated scalar. The first eigenvector and SVD node are then compared, through a difference and a norm of difference. 

In order to see more characteristics of the separated SVD modes, they are ranked by percentage of variance and graphed to show a gradient map.

### Theoretical Background:
#### Pearson Correlation Coefficient:
Pearson Correlation Coefficient (PCC), otherwise known as the Pearson Product-moment Correlation Coefficient (PPECC), is a measure of linear correlation between two sets of data denoted by this formula: 

$r = \frac{\Sigma(x_i - \bar{x})}{\sqrt{\Sigma (x_i - \bar{x})^2 \Sigma (y_i - \bar{y})^2}}$

x and y are the two parts of the matrix X and r is the resulting correlation matrix

#### Eigenvectors and Eigenvalues:
The eigenvector is the characteristic vector of a matrix, or the vector that changes the most during a linear transformation. The corresponding eigenvalue is the factor by which the eigenvector is scaled. For the sake of this lab, we will be using eigenvectors to reduce the dimensionality of the data, from 1,024 ✕ 1,024, down to a series of vectors with 1,024 direction components.

#### Singular Value Decomposition:
Singular Value Decomposition (SVD) is the factorization of a 2D matrix of m ✕ n into 3 component parts: a unitary matrix U that is m ✕ m, a diagonal matrix $\Sigma$ that is scaled by a factor of the 1D matrix S, and the transpose of unitary n ✕ n matrix V. The transpose of V is the principal component direction of the data. Within machine learning, SVD is used to simplify complex data that is not completely square like what is done with eigenvectors.

$A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V^T_{n \times n}$


#### Norm:
The norm is the magnitude of a vector. That is done by taking the summation of the magnitude of each vector part, exponent p and then taking the p root of all of it, as shown by this formula:

### Algorithm Interpretation and Development:
#### numpy.corrcoef(x):
A function that inputs the 2D matrix x and returns a correlation coefficient matrix of the variables done through Pearson correlation.

#### scipy.linalg.eig(x):
A function that inputs a square 2D matrix x and returns the eigenvalues w and their corresponding eigenvectors v ranked by the size of the eigenvalues (greatest to least).

#### scipy.linalg.svd(x):
A function that inputs a 2D matrix x and returns the unitary matrix U, a vector S of scalars used for the diagonal matrix sigma, and the transpose of unitary matrix V. It is ranked by the size of S (greatest to least). 

### Computational Results:
#### Most and Least Correlation:
Based on the first 100 values in comparison to one another, these are the results of most correlated and least correlated.
![download (1)](https://user-images.githubusercontent.com/66970342/232255277-e806bdd1-a0a3-4eaa-be6c-b5a0e04c4c77.png)
![download (2)](https://user-images.githubusercontent.com/66970342/232255283-181e6118-0127-4205-ace5-734db189847d.png)


Based on the selected values in comparison to one another, these are the results of the most correlated and the least correlated. 
![download (4)](https://user-images.githubusercontent.com/66970342/232255289-7e2dc5c9-d979-4e92-a1c0-6957ae38bc87.png)
![download (5)](https://user-images.githubusercontent.com/66970342/232255295-9b6098c1-68a1-4b20-bf24-9d904f2702c4.png)

#### Eigenvectors and Eigenvalues:
These are the top 6 eigenvectors and their corresponding eigenvalues.

```
Eigenvector 1: [ 2.38432673e-02  4.53537771e-02  5.65319581e-02 ...  2.18059778e-02
 -1.34401900e-02 -8.35866572e-05]
Eigenvalue 1: (234020.4548538858+0j)
Eigenvector 2: [ 0.02576146  0.04567536  0.04709124 ... -0.03147167  0.01696919
 -0.0065104 ]
Eigenvalue 2: (49038.31530059219+0j)
Eigenvector 3: [ 0.02728448  0.04474528  0.0362807  ...  0.02274935 -0.01681751
  0.00353767]
Eigenvalue 3: (8236.539897013154+0j)
Eigenvector 4: [ 0.0289902   0.04316163  0.02344727 ... -0.00888087  0.01593233
  0.0004846 ]
Eigenvalue 4: (6024.871457930158+0j)
Eigenvector 5: [ 0.03057294  0.04080838  0.00992662 ...  0.00478419 -0.01159475
 -0.00296729]
Eigenvalue 5: (2051.496432691053+0j)
Eigenvector 6: [ 0.03229324  0.03805116 -0.00241627 ... -0.00896498  0.02049875
  0.00495589]
Eigenvalue 6: (1901.079114823661+0j)
```
#### SVD and Principal Component Directions:
These are the top 6 principal component directions.

```
Principal Component Directions: 
[[-0.01219331 -0.00215188 -0.01056679 ... -0.02177117 -0.03015309
  -0.0257889 ]
 [ 0.01938848  0.00195186 -0.02471869 ... -0.04027773 -0.00219562
  -0.01553129]
 [-0.01691206 -0.00143586 -0.0384465  ... -0.01340245  0.01883373
  -0.00643709]
 [-0.0204079   0.01201431 -0.00397553 ...  0.01641295  0.04011563
  -0.02679029]
 [ 0.01902342 -0.00418948 -0.0384026  ...  0.01092512 -0.00087341
  -0.01260435]
 [ 0.0090084   0.00624237 -0.01580824 ...  0.00977639 -0.00090316
  -0.00304479]]
```

By taking the difference, we get this matrix,

```
[ 0.04768653  0.09070755  0.11306392 ...  0.02418675 -0.01185159
 -0.00049382]
```

But by taking the norm of the absolute value of the differences, we get this singular value, which shows the difference between the eigenvector and the principal component direction.

```
0.5673500824225995
```

These are the 6 SVD modes and their percentage of variance ranked

```
[0.77677271 0.16277049 0.02733915 0.01999806 0.00680943 0.00631016]
```

These are what the first 6 SVD modes look like. 
![download](https://user-images.githubusercontent.com/66970342/232255323-a003ff93-3559-45d3-8a73-ff980ace1846.png)


### Summary and Conclusions:
Through this lab, we saw different ways that the computer can be able to process more complex data. In the first experiment, a basic comparison matrix was used to determine the images that have the most similar traits or the least similar traits from one another. Then, we learned about how characteristics can be simplified and pronounced by the use of eigenvectors, or characteristic vectors; for situations where data isn’t square like ours, SVD could be used instead which, as shown by the comparison, is not super divergent from the eigenvectors. 

Data is very complex, especially in the realms of image processing, and to simplify it, machine learning processes find methods to correlate the data to simplify or categorize the data into more digestible pieces. Because we aren’t instructing machine learning processes about how to associate and divide the data in unsupervised learning, there is a potential risk for biases and mistakes to show up implicitly especially in facial recognition. Because of the biases with complex data, it is very important to make sure that the data can be reduced or simplified to only what is relevant for a specific machine learning task. 
