import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load the datasets
train_df = pd.read_csv('widsdatathon2024-challenge1/training.csv')
test_df = pd.read_csv('widsdatathon2024-challenge1/test.csv')

# Identify numerical and categorical columns in train_df, excluding the target variable
train_numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.drop(['DiagPeriodL90D'], errors='ignore') # errors='ignore' allows for the column to be missing
train_categorical_columns = train_df.select_dtypes(include=['object']).columns


# Initialize the imputers
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
simple_imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputers on the training data
train_df[train_numerical_columns] = knn_imputer.fit_transform(train_df[train_numerical_columns])
train_df[train_categorical_columns] = simple_imputer.fit_transform(train_df[train_categorical_columns])

# Apply the fitted imputers to the test data (transform only, no fitting)
test_df[train_numerical_columns] = knn_imputer.transform(test_df[train_numerical_columns])
test_df[train_categorical_columns] = simple_imputer.transform(test_df[train_categorical_columns])

# Prepare the features and target variable for training
X = train_df.drop(['patient_id', 'DiagPeriodL90D'], axis=1)
y = train_df['DiagPeriodL90D']

# Prepare the test set features
X_test = test_df.drop('patient_id', axis=1)

# Preprocessing steps
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Pipeline for the regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# Train the model
model.fit(X, y)

# Predict on the test set
y_pred = model.predict(X_test)

# Prepare the submission DataFrame
submission_df = pd.DataFrame({
    'patient_id': test_df['patient_id'].astype(int),
    'DiagPeriodL90D': y_pred
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)
