# Used Car Price Prediction 

Assignment 5 for Epochs '26 - building regression models to predict the selling price of used cars using the CarDekho dataset.

## Business Objective

When someone lists a used car on a platform like CarDekho, figuring out a fair asking price is hard - price it too high and it won't sell, too low and the seller loses money. The goal of this project is to build a model that predicts a car's selling price based on its specs (brand, age, mileage, engine size, fuel type, etc.), so both buyers and sellers have a data-backed reference point instead of guessing.

## Dataset Overview

- **Source:** [CarDekho Used Car Dataset](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data) (Kaggle)
- **Original size:** 15,411 rows × 14 columns
- No missing values in any column
- After removing extreme price outliers (very high-end luxury listings, using the IQR method) the working dataset is **14,679 rows**
- Covers 32 different car brands, from Maruti and Hyundai to BMW and Land Rover

## Features and Target Variable

**Target:** `selling_price`

**Numerical features:** `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`

**Categorical features:** `brand`, `seller_type`, `fuel_type`, `transmission_type`

(`car_name` and `model` were dropped - `model` alone had ~120 unique values which is too many to one-hot encode cleanly, and `brand` already captures most of that signal.)

## Preprocessing

- Dropped irrelevant/high-cardinality columns (`Unnamed: 0`, `car_name`, `model`)
- Removed extreme outliers in `selling_price` using the IQR method
- One-hot encoded the categorical columns (`drop_first=True` to avoid the dummy variable trap)
- Split the data 80/20 into train and test sets (`random_state=42`)
- Applied `StandardScaler` to all features - mainly needed for Linear Regression since it's sensitive to feature scale, but applied consistently across all three models to keep the comparison fair

## Regression Models Implemented

1. **Linear Regression** - baseline model
2. **Decision Tree Regressor** (`max_depth=10`)
3. **Random Forest Regressor** (`n_estimators=200`, `max_depth=12`)

## Performance Comparison

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 115,168 | 2.70e+10 | 164,284 | 0.804 |
| Decision Tree | 85,086 | 1.73e+10 | 131,596 | 0.874 |
| **Random Forest** | **74,885** | **1.26e+10** | **112,442** | **0.908** |

## Best Performing Model: Random Forest Regressor

Random Forest came out on top across every metric - highest R² (0.908) and the lowest MAE/RMSE. It performed better than:

- **Linear Regression**, because the relationship between price and features like age or km driven isn't actually a straight line - cars lose value fast in the first few years, then level off, and linear regression can't model that curve.
- **Decision Tree**, because a single tree tends to overfit to quirks in the training data. Random Forest averages many trees together, which smooths that out and generalizes better.

The top features driving predictions were `max_power`, `engine`, `vehicle_age`, and `km_driven` — which lines up with how people actually evaluate a used car in real life.

## Key Observations

- `selling_price` is heavily right-skewed (a small number of very expensive cars pull the average up), which hurts linear models more than tree-based ones
- Tree-based ensemble methods handled the non-linear pricing patterns noticeably better than a straight-line model
- Feature scaling doesn't really matter for tree-based models but was still applied for consistency
- No missing data made preprocessing simpler than usual - most of the effort went into handling high-cardinality categorical features and price outliers

## Future Improvements

1. **Hyperparameter tuning** - use `GridSearchCV`/`RandomizedSearchCV` to properly tune the Random Forest instead of manually picked values
2. **Try gradient boosting** - XGBoost or LightGBM typically beat Random Forest on tabular data like this
3. **Smarter handling of the `model` column** - target encoding or grouping rare models into an "Other" bucket instead of dropping it entirely
4. **Log-transform the target** - training on `log(selling_price)` could help Linear Regression handle the skew better

## Files

- `car_price_prediction.ipynb` - full workflow: EDA, preprocessing, model training, evaluation
- `README.md` - this file

---
Submitted for **#evn-ds-epochs26-day05**
