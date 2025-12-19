import pandas as pd
import numpy as np
import importlib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error


CATEGORICAL_COLS = ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']
NUMERIC_COLS_ALL = [
    'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
    'Crop_Yield_MT_per_HA', 'Extreme_Weather_Events',
    'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA',
    'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index',
    'Economic_Impact_Million_USD'
]
NON_SCALED_EXCLUSIONS = ['Year', 'Extreme_Weather_Events']


def train_and_predict(df, target, model_key, excel_path, input_row,memberName, return_encoder=False):
    import importlib
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, mean_squared_error
    from core.HistoryTraining import CopyAtrainingHistoryFrom
    from config import RAW_DATA_PATH


    # Load model info
    models_df = pd.read_excel(excel_path)
    row = models_df[models_df["Name to be used as a parameter"] == model_key].iloc[0]
    module, cls = row["ModelClassPath"].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module), cls)

    params = {}
    param_str = row.get("parameter Default", "")
    if isinstance(param_str, str) and param_str.strip():
        try:
            params = json.loads(param_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    # Features and target
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Detect classification
    is_classification = any(suffix in model_key.upper() for suffix in [
        "_CLS", "LOGISTIC", "SVM_CLS", "KNN_CLS", "RANDOM_FOREST_CLS",
        "EXTRA_TREES_CLS", "GRADIENT_BOOSTING_CLS", "LIGHTGBM_CLS", "CATBOOST_CLS", "XGBOOST_CLS"
    ])

    le_target = False
    if is_classification:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    # Encode categorical features in X
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

        # Encode input_row if it contains this column
        if col in input_row.columns:
            val = str(input_row[col].iloc[0])
            if val in le.classes_:
                input_row[col] = le.transform([val])
            else:
                input_row[col] = [0]  # fallback for unseen category

    # Numeric features
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = X[num_cols].astype(float)
    input_row[num_cols] = input_row[num_cols].astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    input_row[num_cols] = scaler.transform(input_row[num_cols])

    # Train model
    pipeline = Pipeline([("model", ModelClass(**params))])
    pipeline.fit(X_train, y_train)

    # Predict
    prediction = pipeline.predict(input_row)[0]

    # Compute metric
    if is_classification:
        score = accuracy_score(y_test, pipeline.predict(X_test))
        metric = "Accuracy"
    else:
        score = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
        metric = "RMSE"
    raw_df = pd.read_csv(RAW_DATA_PATH)
    CopyAtrainingHistoryFrom(
        model_key,
        pipeline,
        memberName,
        target,
        (1-score)*0.8 if is_classification else score,
        (
            (raw_df.loc[df.index[df[target] == prediction].tolist()[0], target] if df.index[df[target] == prediction].tolist() else "Unknown")
            if is_classification else
            str(round(prediction * (raw_df[target].std() if target in raw_df else 1) + (raw_df[target].mean() if target in raw_df else 0), 2))
        ),
        ("classification" if is_classification else "regression")
    )
    return (prediction, (1-score)*0.8, metric, le_target) if is_classification and return_encoder else (prediction, score, metric,None)
