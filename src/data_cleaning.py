# Use pandas, os libraries

# Read 'data/raw/export.csv' into a pandas DataFrame and print initial row/column count.

# Drop any rows where 'facility_rating_status' is NaN (unlabelled records cannot be used for training).

# Create a 'target' column:
#   - 0 = 'Pass' (Compliant / Low Risk), 1 = 'Conditional Pass' or 'Closure' (High Risk / Non-Pass)

# Select key features: permit_type, neighborhood', total inspection time, inspection frequency

# Fill missing numeric values using column medians, and text values with 'Unknown'.

# Convert text columns into binary dummy variables using pd.get_dummies(drop_first=True).

# Combine features (X) and target (Y), then export to 'data/processed/cleaned_data.csv'.