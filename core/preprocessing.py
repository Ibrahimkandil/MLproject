import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Store encoders in a dictionary
encoders = {
    'Country': LabelEncoder(),
    'Region': LabelEncoder(),
    'Crop_Type': LabelEncoder(),
    'Adaptation_Strategies': LabelEncoder()
}
scaler = StandardScaler()

numeric_cols = [
    'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
    'Crop_Yield_MT_per_HA', 'Extreme_Weather_Events',
    'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
    'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index',
    'Economic_Impact_Million_USD'
]

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)

    # 1. Fill NaNs
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # 2. Fit and Transform Categorical (One encoder per column)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 3. Fit and Transform Numeric
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.to_csv(output_path, index=False)
    return df
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# scaler = StandardScaler()
# encoder = LabelEncoder()
#
#
#
# def clean_dataset(input_path, output_path):
#     df = pd.read_csv(input_path)
#
#     df = df.fillna({
#         'Average_Temperature_C': df['Average_Temperature_C'].mean(),
#         'Total_Precipitation_mm': df['Total_Precipitation_mm'].mean(),
#         'CO2_Emissions_MT': df['CO2_Emissions_MT'].mean(),
#         'Crop_Yield_MT_per_HA': df['Crop_Yield_MT_per_HA'].mean(),
#         'Soil_Health_Index': df['Soil_Health_Index'].mean(),
#         'Economic_Impact_Million_USD': df['Economic_Impact_Million_USD'].mean(),
#     })
#
#     for col in ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']:
#         if col in df.columns:
#             df[col] = encoder.fit_transform(df[col].astype(str))
#
#     numeric_cols = [
#     'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
#     'Crop_Yield_MT_per_HA', 'Extreme_Weather_Events',
#     'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
#     'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index',
#     'Economic_Impact_Million_USD'
#     ]
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#
#     df.to_csv(output_path, index=False)
#     return df
