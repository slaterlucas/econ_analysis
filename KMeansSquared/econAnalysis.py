import os
import warnings
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas_datareader import wb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress FutureWarnings temporarily
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the indicators and countries
indicators = {
    'GDP_Per_Capita': 'NY.GDP.PCAP.CD',
    'CO2_Emissions': 'EN.ATM.CO2E.PC',  # Update this if necessary
    'Forest_Area': 'AG.LND.FRST.ZS',
    'Access_to_Clean_Water': 'SH.H2O.BASW.ZS'  # Updated code
}

countries = ['USA', 'CHN', 'IND', 'BRA', 'RUS', 'ZAF', 'MEX', 'IDN', 'NGA', 'DEU']  # A diverse set of countries

# Fetch data for the past 20 years
start_year = 2000
end_year = 2019

# Function to list available indicators (optional, for verification)
def list_available_indicators(keyword='CO2'):
    indicators_df = wb.get_indicators()
    filtered = indicators_df[indicators_df['name'].str.contains(keyword, case=False, na=False)]
    print(filtered[['id', 'name']])

# Uncomment to list CO2-related indicators
# list_available_indicators()

# Fetch data from the World Bank using wb.download
data_frames = []
for indicator_name, indicator_code in indicators.items():
    try:
        df = wb.download(
            indicator=indicator_code,
            country=countries,
            start=start_year,
            end=end_year
        )
        if df.empty:
            print(f"No data returned for indicator: {indicator_name} ({indicator_code})")
            continue  # Skip to the next indicator
        df = df.reset_index()
        df = df.rename(columns={indicator_code: indicator_name})
        data_frames.append(df[['country', 'year', indicator_name]])
    except Exception as e:
        print(f"Error fetching data for {indicator_name} ({indicator_code}): {e}")

# Check if any data was fetched
if not data_frames:
    raise ValueError("No data was fetched for any of the indicators. Please check indicator codes and parameters.")

# Merge all data frames on 'country' and 'year'
from functools import reduce
try:
    data = reduce(lambda left, right: pd.merge(left, right, on=['country', 'year']), data_frames)
except Exception as e:
    print(f"Error merging data frames: {e}")
    raise

# Drop rows with missing values
data = data.dropna()

# Check if data is sufficient
if data.empty:
    raise ValueError("Merged data is empty after dropping missing values.")

# Define features and target variable
X = data[['Access_to_Clean_Water', 'CO2_Emissions', 'Forest_Area']]
y = data['GDP_Per_Capita']

# Check for sufficient data
if X.empty or y.empty:
    raise ValueError("Feature set or target variable is empty.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = rf.feature_importances_

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Ensure the output directory exists
output_dir = 'output/plots'
os.makedirs(output_dir, exist_ok=True)

# Visualize feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Set1')
plt.title('Feature Importance - Environmental Factors on Economic Prosperity')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Save plot to the desired directory
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.show()

# Display the importance DataFrame
print(importance_df)
