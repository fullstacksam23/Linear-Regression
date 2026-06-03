# Linear Regression Notebook Summary

This project builds a multiple linear regression model to predict house prices using the dataset in `Housing.csv`, as documented in `linear-regression.ipynb`.

The notebook starts with basic data inspection using `head()`, `describe()`, and `info()` to understand the 545-row dataset and identify numerical, boolean, and categorical features. It also checks for missing values and confirms that no imputation is needed.

Preprocessing includes:

- Mapping boolean-style columns such as `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, and `prefarea` from `yes/no` to `1/0`
- One-hot encoding `furnishingstatus` with `drop_first=True`
- Splitting the data into features (`X`) and target (`price`)
- Creating training and test sets with an 80/20 split using `random_state=42`
- Standardizing numerical columns (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`) with `StandardScaler`, fit only on the training set to avoid data leakage

After preprocessing, the notebook trains a `LinearRegression` model from scikit-learn and evaluates it on the test set. The reported results are:

- MAE: `970,043.40`
- MSE: `1,754,318,687,330.67`
- R-squared: `0.6529`

These results indicate that the model explains about `65.29%` of the variation in house prices, with an average prediction error of roughly `INR 970,043`.

The notebook also includes visual analysis:

- A scatter plot comparing actual vs predicted prices
- A residual plot to inspect prediction errors
- A bar chart of regression coefficients

The coefficient analysis shows that `airconditioning`, `hotwaterheating`, and `prefarea` have the strongest positive impact on predicted price, while `unfurnished` has a negative effect relative to the baseline furnishing category. Structural features such as `area`, `bathrooms`, and `stories` also contribute meaningfully, while `bedrooms` has a comparatively smaller effect once other variables are included.
