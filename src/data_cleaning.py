import pandas as pd
import os

# Read in the raw csv file, drop rows with missing 'facility_rating_status', and create a target column of binary values.
def load_and_filter(file_path: str) -> pd.DataFrame:

    # Load the CSV file into a DataFrame
    print(f"\nLoading raw CSV data from {file_path}")
    df = pd.read_csv(file_path)
    initial_row_count, initial_col_count = df.shape[0], df.shape[1]

    # Drop rows with missing 'facility_rating_status' and create a target column of binary values.
    df = df.dropna(subset=['facility_rating_status']).copy()
    df['target'] = df['facility_rating_status'].apply(lambda x: 0 if x == 'Pass' else 1)

    filtered_row_count, filtered_col_count = df.shape[0], df.shape[1]

    # Display how shape of dataframe has changed after filtering
    print(f"Initial row and column count: rows={initial_row_count}, columns={initial_col_count}")
    print(f"Filtered row and column count: rows={filtered_row_count}, columns={filtered_col_count}")

    return df

# Select which features to include, clean formatting, impute missing values
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Features which have potential to be risk predictors for passing or failing inspection
    keep_features = ['target', 'permit_type', 'analysis_neighborhood', 'inspection_frequency_type', 'inspection_type', 'total_time']
    df_keep = df[keep_features].copy()    

    # Helper function to help categorize the many kinds of 'permit_type'
    def categorize_permit(permit: str) -> str:
        permit = str(permit).upper()
        if "RESTAURANT" in permit:
            return "Restaurant"
        elif "RETAIL MKT" in permit or "SUPERMARKET" in permit:
            return "Retail Market"
        elif "BAR" in permit or "TAVERN" in permit:
            return "Bar/Tavern"
        elif "CAFETERIA" in permit:
            return "Cafeteria"
        elif "SUMMER MEALS" in permit or "COMMUNITY EVENT" in permit:
            return "Community/Summer Meals"
        elif "MOBILE" in permit or "PUSHCART" in permit:
            return "Mobile Food"
        elif "WHOLESALE" in permit or "MANUFACTURING" in permit:
            return "Wholesale/Manufacturing"
        else:
            return "Other"

    # Clean the 'permit_type' column by categorizing its values
    df_keep["permit_type"] = df_keep["permit_type"].apply(categorize_permit)

    # Clean total_time column:convert to float, take absolute values and impute missing values with the median
    df_keep['total_time'] = pd.to_numeric(df_keep['total_time'], errors='coerce')
    df_keep['total_time'] = df_keep['total_time'].abs()
    median_total_time = df_keep['total_time'].median()
    df_keep['total_time'] = df_keep['total_time'].fillna(median_total_time)

    # Clean the categorical features and fill missing values with 'Unknown'
    categorical_features = ['permit_type', 'analysis_neighborhood', 'inspection_frequency_type', 'inspection_type']
    for feature in categorical_features:
        df_keep[feature] = df_keep[feature].fillna('Unknown').astype(str)

    print(f'\nFeature Cleaning and Engineering Completed. Columns Chosen: {df_keep.columns.tolist()}')
    print(f"Permit type categories:\n{df_keep['permit_type'].value_counts()}")
    print(f"\nMissing values remaining: {df_keep.isnull().sum().sum()}")

    return df_keep

# Export a cleaned datset for EDA, and one for modeling
def export_datasets(df: pd.DataFrame, output_path: str) -> None:
    # Export EDA dataset
    os.makedirs(output_path, exist_ok=True)
    eda_data_path = os.path.join(output_path, "cleaned_data_eda.csv")
    df.to_csv(eda_data_path, index=False)
    print(f'EDA dataset exported to: {eda_data_path} shape: {df.shape}')

    # Convert categorical columns to dummy variables for modeling
    categorical_features = ['permit_type', 'analysis_neighborhood', 'inspection_frequency_type', 'inspection_type']
    df_modeling = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Export modeling dataset
    modeling_data_path = os.path.join(output_path, "cleaned_data_modeling.csv")
    df_modeling.to_csv(modeling_data_path, index=False)
    print(f'Modeling dataset exported to: {modeling_data_path} shape: {df_modeling.shape}')


if __name__ == "__main__":
    file_path = os.path.join("data", "raw-data", "export.csv")
    cleaned_df = load_and_filter(file_path)
    engineered_df = feature_engineering(cleaned_df)
    export_datasets(engineered_df, os.path.join("data", "cleaned-data"))





