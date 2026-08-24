# NITJ_MFAI
# Python Covariance Matrix Solver

Python script to compute the covariance matrix of a user-provided dataset. It calculates the statistical covariance between features from scratch without relying on external libraries like NumPy or Pandas.

## Features
* Built entirely using native Python loops and data structures.
* Accepts custom datasets directly from the terminal.
* Ensures the user inputs the correct number of features and valid numeric data types.
* Displays the final covariance matrix rounded nicely to 4 decimal places.

## How it Works
1. **Mean Calculation**: Computes the arithmetic mean for each feature (column) across all samples.
2. **Covariance Computation**: Calculates the sample covariance between every pair of features using the formula:
   $$\text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{n - 1}$$
3. **Matrix Construction**: Assembles the values into an $N \times N$ matrix, where $N$ is the number of features.

### Example Run
```text
Enter number of rows (sample): 6
Enter number of columns (feature): 2
Enter 2 values for each row: 
Row 1: 1.70 72
Row 2: 1.62 64
Row 3: 1.52 84
Row 4: 1.85 80
Row 5: 1.91 72
Row 6: 1.42 70

Covariance Matrix is :
['0.0357', '0.1080']
['0.1080', '51.8667']
```
