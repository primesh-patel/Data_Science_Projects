# Data Preprocessing Pipeline

## Project Overview
The objective of this project is to develop a data preprocessing pipeline that cleans and standardizes raw datasets by handling missing values, detecting and managing outliers, and normalizing numeric features. The final preprocessed dataset is then ready for further analysis or machine learning tasks.

## Steps in the Data Preprocessing Pipeline

### Step 1: Identify Numeric and Categorical Features
- Numeric Features: `NumericFeature1`, `NumericFeature2`
- Categorical Features: `CategoricalFeature`

### Step 2: Handling Missing Values in Numeric Features
- Missing values in numeric features are filled with the mean of the respective column.

### Step 3: Outlier Detection and Handling
- Outliers in numeric features are identified using the Interquartile Range (IQR) method.
- Any value outside 1.5 times the IQR below Q1 or above Q3 is replaced with the mean of the column.

### Step 4: Normalization of Numeric Features
- Numeric features are standardized using `StandardScaler` to ensure all values are on a comparable scale.

### Step 5: Handling Missing Values in Categorical Features
- If a categorical column has missing values, they are replaced with the most frequent value (mode).
- If an entire column contains missing values, they are replaced with a default value (`Unknown`).

## Input Dataset Example
| NumericFeature1 | NumericFeature2 | CategoricalFeature |
|----------------|----------------|-------------------|
| 1.0 | 7 | A |
| 2.0 | 8 | B |
| NaN | 9 | NaN |
| 4.0 | 10 | A |
| 5.0 | 11 | B |

## Output After Preprocessing
| NumericFeature1 | NumericFeature2 | CategoricalFeature |
|----------------|----------------|-------------------|
| -1.535624 | -1.099370 | A |
| -0.944999 | -0.749128 | B |
| 0.000000 | -0.398886 | A |
| 0.236250 | -0.048645 | A |
| 0.826874 | 0.301597 | B |
| 1.417499 | 1.994431 | C |

## Conclusion
The preprocessing pipeline successfully cleans the dataset by handling missing values, outliers, and standardizing numeric features, making it suitable for further data analysis and machine learning applications. This ensures data consistency and improves the quality of the input for downstream processes.

