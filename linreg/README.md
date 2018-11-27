# Linear Regression
This subdirectory contains a CUDA-based implementation of the Linear Regression algorithm.

## Setup
This project assumes that you have `nvcc` installed on your machine.
To compile the source code run make, this will create an executable in the obj folder called linreg, as well as a library file called linreg.source

## Running
The linreg can be run by navigating to the obj directory and executing ./linreg which performs linear regression on the data set defined in house_med.h
This header file can be changed to allow the program to operate on other data sets, or the individual functions in the file can be called to fit and predict data from elsewhere
The program will run three tests on increasingly large sections of the data set that will each be timed

## Output
Each test will output the time of the test, the coeffcient vector geneated by the linear regression, the predicted vs actual values, the average error, and the RMSE
