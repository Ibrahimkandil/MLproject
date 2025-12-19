import pandas as pd

def crop_by_country(df):
    return df.groupby(["Country", "Crop_Type"]).size().reset_index(name="count")

def impact_over_time(df):
    return df.groupby(["Year", "Country"])["Economic_Impact_Million_USD"].mean().reset_index()
