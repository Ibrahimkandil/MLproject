import requests
from flask import Flask
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from core.trainer import train_and_predict
from datetime import datetime
import pandas as pd
from config import RAW_DATA_PATH, CLEAN_DATA_PATH, MODELS_EXCEL
from core.preprocessing import clean_dataset
from core.preprocessing import encoders, scaler, numeric_cols
from core.model_registry import MEMBER_TARGET_MODELS
from core.HistoryTraining import initialize,GetHistory
from flask import request, jsonify
from geopy.geocoders import Nominatim
from flask import Flask, render_template
import numpy as np
app = Flask(__name__)

initialize()
df = clean_dataset(RAW_DATA_PATH, CLEAN_DATA_PATH)
df_models = pd.read_excel(MODELS_EXCEL)
ALL_MODELS = df_models["Name to be used as a parameter"].tolist()
raw_df = pd.read_csv(RAW_DATA_PATH)
cleaned_df = pd.read_csv(CLEAN_DATA_PATH)
COLUMNS = raw_df.columns.tolist()
AVAILABLE_COUNTRIES = sorted(raw_df["Country"].dropna().unique().tolist())
CURRENT_YEAR = datetime.now().year
raw_df = pd.read_csv(RAW_DATA_PATH)
TARGETS = list(MEMBER_TARGET_MODELS.keys())
COUNTRIES = sorted(raw_df["Country"].dropna().unique())
ALLOWED_TARGETS = ["Crop_Type", "Economic_Impact_Million_USD", "Adaptation_Strategies"]
ResultHistory=[]
raw_df = pd.read_csv(RAW_DATA_PATH)
categorical_cols = ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()
mean_row = pd.DataFrame({col: [df[col].mean()] for col in numeric_cols})
mode_row = pd.DataFrame({col: [df[col].mode()[0]] for col in categorical_cols if col in df.columns})
input_row_full = pd.concat([mean_row, mode_row], axis=1)
df = clean_dataset(RAW_DATA_PATH, CLEAN_DATA_PATH)  # your existing function
# COUNTRY_COORDS = {
#     "Tunisia":[33.8869,9.5375], "France":[46.2276,2.2137], "USA":[37.0902,-95.7129],
#     "Brazil":[-14.2350,-51.9253], "India":[20.5937,78.9629], "Germany":[51.1657,10.4515],
#     "Italy":[41.8719,12.5674], "Spain":[40.4637,-3.7492], "Canada":[56.1304,-106.3468],
#     "China":[35.8617,104.1954]
# }
COUNTRY_COORDS = {"Tunisia":[33.8869,9.5375],"Canada":[56.1304,-106.3468],
    "Brazil":[-14.2350,-51.9253], "India":[20.5937,78.9629]
}
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
input_row_full = pd.DataFrame({col: [df[col].mean()] for col in numeric_cols})
def build_reverse_mapping(raw_df, cleaned_df, categorical_columns):
    reverse_maps = {}

    for col in categorical_columns:
        if col in raw_df.columns and col in cleaned_df.columns:
            mapping = {}
            for raw_val, clean_val in zip(raw_df[col], cleaned_df[col]):
                if clean_val not in mapping:
                    mapping[clean_val] = raw_val
            reverse_maps[col] = mapping

    return reverse_maps
REVERSE_MAP = build_reverse_mapping(raw_df, df, categorical_cols)
@app.route("/")
def home():
    facts = raw_df.sample(5).to_dict(orient="records")
    return render_template("home.html", facts=facts)
def generate_eda_html(df):
    html = '<div class="row">'
    for col in df.columns:
        html += '<div class="col-md-6 mb-4"><div class="card"><div class="card-body">'
        html += f'<h5 class="card-title">{col}</h5>'

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe().to_frame().round(2)
            html += desc.to_html(classes="table table-sm table-striped")

            plt.figure()
            sns.histplot(df[col].dropna(), kde=True)
        else:
            counts = df[col].value_counts()
            html += counts.to_frame().to_html(classes="table table-sm table-striped")

            plt.figure()
            counts.plot(kind="bar")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()

        img_base64 = base64.b64encode(buf.getvalue()).decode()
        html += f'<img src="data:image/png;base64,{img_base64}" class="img-fluid mt-2"/>'

        html += '</div></div></div>'
    html += '</div>'
    return html
@app.route("/visualize")
def visualize():
    profile_html = generate_eda_html(raw_df)

    crop_counts = df["Crop_Type"].value_counts()
    crop_chart = go.Figure([go.Bar(x=crop_counts.index, y=crop_counts.values)])

    impact_chart = go.Figure([
        go.Scatter(x=df["Year"], y=df["Economic_Impact_Million_USD"], mode="lines+markers")
    ])

    return render_template(
        "visualize.html",
        profile_html=profile_html,
        crop_chart=json.dumps(crop_chart, cls=plotly.utils.PlotlyJSONEncoder),
        impact_chart=json.dumps(impact_chart, cls=plotly.utils.PlotlyJSONEncoder)
    )
def prepare_numeric_input(raw_df, cleaned_df, form, target):
    input_data = {}

    # Step 1: Collect input from form
    for col in raw_df.columns:
        if col == target:
            continue

        val = form.get(col, "").strip()

        if val == "":
            # Fill missing values
            if np.issubdtype(raw_df[col].dtype, np.number):
                input_data[col] = raw_df[col].mean()
            else:
                input_data[col] = raw_df[col].mode()[0]
        else:
            # Try converting to float; keep string if fails (categorical)
            try:
                input_data[col] = float(val)
            except ValueError:
                input_data[col] = val

    # Step 2: Convert any categorical value to numeric using raw_df -> cleaned_df mapping
    numeric_input_df = pd.DataFrame([input_data])
    for col in numeric_input_df.columns:
        val = numeric_input_df[col].iloc[0]
        if isinstance(val, str):
            found = False
            for raw_col in raw_df.columns:
                if str(val) in raw_df[raw_col].astype(str).values:
                    row_index = raw_df[raw_col].astype(str) == str(val)
                    if raw_col in cleaned_df.columns:
                        numeric_input_df[col] = cleaned_df.loc[row_index, raw_col].values[0]
                        found = True
                        break
            if not found:
                # fallback to numeric mean if mapping fails
                numeric_input_df[col] = cleaned_df[col].mean() if col in cleaned_df.columns else 0

    # Step 3: Only convert original numeric columns to float
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_input_df.columns:
        if col in numeric_cols:
            numeric_input_df[col] = numeric_input_df[col].astype(float)

    # Step 4: Keep only training columns
    train_columns = cleaned_df.drop(columns=[target]).columns.unique()
    numeric_input_df = numeric_input_df.loc[:, numeric_input_df.columns.intersection(train_columns)]

    return numeric_input_df
@app.route("/api/autofill")
def autofill():
    try:
        country = request.args.get("country")
        year = request.args.get("year", type=int)

        if not country or not year:
            return jsonify({"error": "Country and year required"}), 400

        country_df = raw_df[raw_df["Country"] == country]

        # Safe means/modes
        means = {
            "Region": country_df["Region"].mode()[0] if not country_df.empty and not country_df["Region"].mode().empty else "Unknown",
            "CO2_Emissions_MT": float(country_df["CO2_Emissions_MT"].mean()) if not country_df.empty else 10.0,
            "Crop_Yield_MT_per_HA": float(country_df["Crop_Yield_MT_per_HA"].mean()) if not country_df.empty else 3.5,
            "Extreme_Weather_Events": float(country_df["Extreme_Weather_Events"].mean()) if not country_df.empty else 2.0,
            "Irrigation_Access_%": float(country_df["Irrigation_Access_%"].mean()) if not country_df.empty else 50.0,
            "Pesticide_Use_KG_per_HA": float(country_df["Pesticide_Use_KG_per_HA"].mean()) if not country_df.empty else 1.0,
            "Fertilizer_Use_KG_per_HA": float(country_df["Fertilizer_Use_KG_per_HA"].mean()) if not country_df.empty else 1.0,
            "Soil_Health_Index": float(country_df["Soil_Health_Index"].mean()) if not country_df.empty else 0.5
        }

        # Optional weather
        lat, lon = COUNTRY_COORDS.get(country, (None, None))
        avg_temp = None
        total_precip = None
        if lat and lon:
            try:
                weather_url = (
                    f"https://archive-api.open-meteo.com/v1/archive"
                    f"?latitude={lat}&longitude={lon}"
                    f"&start_date={year}-01-01&end_date={year}-12-31"
                    "&daily=temperature_2m_mean,precipitation_sum"
                    "&timezone=UTC"
                )
                r = requests.get(weather_url, timeout=10)
                r.raise_for_status()
                weather = r.json()
                avg_temp = float(np.mean(weather["daily"]["temperature_2m_mean"]))
                total_precip = float(np.sum(weather["daily"]["precipitation_sum"]))
            except Exception as e:
                print("Weather API error:", e)

        return jsonify({
            "Year": year,
            "Country": country,
            "Average_Temperature_C": avg_temp,
            "Total_Precipitation_mm": total_precip,
            **means
        })

    except Exception as e:
        print("Autofill failed:", e)
        return jsonify({"error": "Autofill failed"}), 500
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    try:
        if request.method == "POST":
            target = request.form.get("target")

            config = MEMBER_TARGET_MODELS.get(target, {})
            member = config.get("member", "Unknown")
            models = config.get("models", [])

            # Prepare input safely
            try:
                input_row = prepare_numeric_input(raw_df, cleaned_df, request.form, target)
            except Exception as e:
                print("Input preparation failed:", e)
                input_row = pd.DataFrame([{}])  # fallback empty

            best_score = -1e9
            best_result = None

            print("target to Train:"+target)

            for model_key in models:
                try:
                    prediction, score, metric, le_target = train_and_predict(
                        cleaned_df,
                        target,
                        model_key,
                        MODELS_EXCEL,
                        input_row,
                        member,
                        return_encoder=True
                    )
                    print("Model Trained:"+model_key)

                    effective_score = score if metric == "Accuracy" else -score
                    if effective_score > best_score:
                        best_score = effective_score
                        best_result = (prediction, score, metric, model_key, le_target)

                except Exception as e:
                    print(f"Model {model_key} failed: {e}")
                    continue

            # Build result safely even if all models fail
            if best_result:
                prediction, score, metric, model_key, le_target = best_result
                try:
                    if le_target is not None:
                        row_indices = cleaned_df.index[cleaned_df[target] == prediction].tolist()
                        final_prediction = raw_df.loc[row_indices[0], target] if row_indices else "Unknown"
                    else:
                        mean = raw_df[target].mean() if target in raw_df else 0
                        std = raw_df[target].std() if target in raw_df else 1
                        final_prediction = round(prediction * std + mean, 2)
                except Exception as e:
                    print("Prediction transform failed:", e)
                    final_prediction = "Unknown"
                result = {
                    "target": target,
                    "member": member,
                    "model": model_key,
                    "prediction": final_prediction,
                    "score": round(score, 4),
                    "metric": metric
                }
            else:
                # Fallback if all models failed
                result = {
                    "target": target,
                    "member": member,
                    "model": "N/A",
                    "prediction": "No prediction available",
                    "score": 0,
                    "metric": "N/A"
                }

    except Exception as e:
        print("Predict route failed:", e)
        # Fallback if entire POST fails
        result = {
            "target": request.form.get("target", "Unknown"),
            "member": "Unknown",
            "model": "N/A",
            "prediction": "Prediction failed",
            "score": 0,
            "metric": "N/A"
        }

    return render_template(
        "predict.html",
        targets=TARGETS,
        members=MEMBER_TARGET_MODELS,
        countries=COUNTRIES,
        current_year=CURRENT_YEAR,
        columns=raw_df.columns,
        result=result
    )
@app.route("/dashboard")
def dashboard():
    history = GetHistory()
    sample_df = cleaned_df.head(10)  # Your 10 lines of cleaned data

    processed_groups = {}

    for h in history:
        display_key = f"{h['Target']} ({h['Member']})"
        is_cls = h.get("LearningType", "").lower() == "classification"

        # --- Run Prediction on 10 rows ---
        model = h.get("ModelObj")
        analysis_data = {}

        if model and hasattr(model, 'predict'):
            features = sample_df.drop(columns=[h['Target']], errors='ignore')
            preds = model.predict(features)
            actuals = sample_df[h['Target']].tolist() if h['Target'] in sample_df else []

            if is_cls:
                # Basic Confusion Mapping for first 10 rows
                analysis_data = {
                    "labels": ["Correct", "Incorrect"],
                    "values": [sum(1 for p, a in zip(preds, actuals) if str(p) == str(a)),
                               sum(1 for p, a in zip(preds, actuals) if str(p) != str(a))]
                }
            else:
                analysis_data = {
                    "labels": [f"Idx {i}" for i in range(len(preds))],
                    "actual": [round(float(a), 3) for a in actuals],
                    "predicted": [round(float(p), 3) for p in preds]
                }

        h["AnalysisData"] = analysis_data

        # --- Grouping ---
        if display_key not in processed_groups:
            processed_groups[display_key] = {}

        model_name = h["ModelName"]
        if model_name not in processed_groups[display_key]:
            processed_groups[display_key][model_name] = h
        else:
            existing_score = processed_groups[display_key][model_name]["Score"]
            if (is_cls and h["Score"] > existing_score) or (not is_cls and h["Score"] < existing_score):
                processed_groups[display_key][model_name] = h

    # Sorting
    final_data = {}
    for key, m_dict in processed_groups.items():
        m_list = list(m_dict.values())
        final_data[key] = sorted(m_list, key=lambda x: x['Score'], reverse=(m_list[0]['LearningType']=='classification'))

    return render_template("dashboard.html", grouped=final_data)

#@app.route("/dashboard")
# def dashboard():
#     history = GetHistory()
#     # Take 10 lines for the live prediction comparison
#     sample_df = cleaned_df.head(10)
#
#     processed_groups = {}
#
#     for h in history:
#         display_key = f"{h['Target']} ({h['Member']})"
#         model_name = h["ModelName"]
#         is_cls = h.get("LearningType", "").lower() == "classification"
#
#         # --- PREDICTION LOGIC ---
#         model_obj = h.get("ModelObj")
#         analysis_data = {}
#
#         if model_obj and hasattr(model_obj, 'predict'):
#             try:
#                 # Prepare features (drop target if it exists in sample)
#                 features = sample_df.drop(columns=[h['Target']], errors='ignore')
#                 live_preds = model_obj.predict(features)
#                 actuals = sample_df[h['Target']].tolist() if h['Target'] in sample_df else []
#
#                 if is_cls:
#                     # Logic for Confusion Matrix (simplified for chart)
#                     # We show count of correct vs incorrect for these 10 lines
#                     correct = sum(1 for p, a in zip(live_preds, actuals) if str(p) == str(a))
#                     analysis_data = {
#                         "labels": ["Correct", "Incorrect"],
#                         "values": [correct, len(live_preds) - correct]
#                     }
#                 else:
#                     # Logic for Regression Curve (Actual vs Predicted)
#                     analysis_data = {
#                         "labels": [f"Pt {i+1}" for i in range(len(live_preds))],
#                         "actual": actuals,
#                         "predicted": [round(float(p), 4) for p in live_preds]
#                     }
#             except Exception as e:
#                 print(f"Prediction error: {e}")
#                 analysis_data = {"error": "Could not run prediction"}
#
#         h["AnalysisData"] = analysis_data
#
#         # --- GROUPING & BEST MODEL FILTERING ---
#         if display_key not in processed_groups:
#             processed_groups[display_key] = {}
#
#         if model_name not in processed_groups[display_key]:
#             processed_groups[display_key][model_name] = h
#         else:
#             existing_score = processed_groups[display_key][model_name].get("Score", 0)
#             if (is_cls and h["Score"] > existing_score) or (not is_cls and h["Score"] < existing_score):
#                 processed_groups[display_key][model_name] = h
#
#     # Finalize Data
#     final_data = {}
#     for key, models_dict in processed_groups.items():
#         model_list = list(models_dict.values())
#         is_cls = model_list[0].get("LearningType", "").lower() == "classification"
#         model_list.sort(key=lambda x: x['Score'], reverse=is_cls)
#         final_data[key] = model_list
#
#     return render_template("dashboard.html", grouped=final_data)


# @app.route("/dashboard")
# def dashboard():
#     history = GetHistory()  # your function
#
#     processed_groups = {}
#
#     for h in history:
#         display_key = f"{h['Target']} ({h['Member']})"
#         model_name = h["ModelName"]
#         score = h.get("Score", 0)
#         is_cls = h.get("LearningType", "").lower() == "classification"
#
#         # Fix missing fields
#         if "ScoreHistory" not in h or not h["ScoreHistory"]:
#             h["ScoreHistory"] = [score]  # default to current score if empty
#         if "ConfusionMatrix" not in h or not h["ConfusionMatrix"]:
#             h["ConfusionMatrix"] = {"labels": ["None"], "values": [0]}
#
#         if display_key not in processed_groups:
#             processed_groups[display_key] = {}
#
#         # Keep best run per model
#         if model_name not in processed_groups[display_key]:
#             processed_groups[display_key][model_name] = h
#         else:
#             existing_score = processed_groups[display_key][model_name].get("Score", 0)
#             if is_cls:
#                 if score > existing_score:
#                     processed_groups[display_key][model_name] = h
#             else:
#                 if score < existing_score:
#                     processed_groups[display_key][model_name] = h
#
#     # Sort models
#     final_data = {}
#     for display_key, models_dict in processed_groups.items():
#         model_list = list(models_dict.values())
#         is_cls = model_list[0].get("LearningType", "").lower() == "classification"
#         model_list.sort(key=lambda x: (x['Score'], x['CreatedTime']), reverse=is_cls)
#         final_data[display_key] = model_list
#
#     return render_template("dashboard.html", grouped=final_data)

from core.preprocessing import encoders, scaler, numeric_cols

import pandas as pd
from flask import render_template
from core.preprocessing import encoders, scaler, numeric_cols

@app.route("/map")
def map_route():
    mass_predictions = []

    # Load raw data for reference
    raw_df = pd.read_csv(RAW_DATA_PATH)
    training_data = df.copy()  # your cleaned training data

    for country_name, coords in COUNTRY_COORDS.items():
        # --- 1. BUILD INPUT ROW ---
        row_dict = {}

        # If country exists in raw data, take its latest row
        if country_name in raw_df['Country'].values:
            country_row = raw_df[raw_df['Country'] == country_name].iloc[-1]
            for col in numeric_cols:
                row_dict[col] = country_row[col]
            for col in ['Region', 'Crop_Type', 'Adaptation_Strategies']:
                row_dict[col] = country_row[col]
            row_dict['Year'] = country_row['Year'] if 'Year' in country_row else 2024
        else:
            # Unseen country: fallback to global mean/mode
            row_dict['Year'] = raw_df['Year'].max() if 'Year' in raw_df else 2024
            for col in numeric_cols:
                row_dict[col] = raw_df[col].mean()
            for col in ['Region', 'Crop_Type', 'Adaptation_Strategies']:
                row_dict[col] = raw_df[col].mode()[0]

        row_dict['Country'] = country_name

        # Convert to DataFrame
        input_row_raw = pd.DataFrame([row_dict])
        input_row_transformed = input_row_raw.copy()

        # --- 2. APPLY TRANSFORMATIONS ---
        # Encode categorical features
        for col, le in encoders.items():
            if col in input_row_transformed.columns:
                val = str(input_row_transformed[col].iloc[0])
                if val in le.classes_:
                    input_row_transformed[col] = le.transform([val])[0]
                else:
                    input_row_transformed[col] = len(le.classes_)  # unseen categories

        # Scale numeric columns
        cols_to_scale = [c for c in numeric_cols if c in input_row_transformed.columns]
        if cols_to_scale:
            input_row_transformed[cols_to_scale] = scaler.transform(input_row_transformed[cols_to_scale])

        # Reorder columns to match training data
        input_row_transformed = input_row_transformed[[col for col in training_data.columns if col in input_row_transformed.columns]]

        # --- 3. MAKE PREDICTIONS ---
        prediction_entry = {"country": country_name, "lat": float(coords[0]), "lon": float(coords[1]), "results": {}}

        for target, info in MEMBER_TARGET_MODELS.items():
            models = info.get("models", [])
            if not models:
                prediction_entry["results"][target] = None
                continue

            model_key = models[0]

            try:
                # Drop target from input features if present
                input_row_model = input_row_transformed.drop(columns=[target], errors="ignore")
                train_features = [col for col in training_data.columns if col != target]
                input_row_model = input_row_model[train_features]

                prediction, score, metric, _ = train_and_predict(
                    df=training_data,
                    target=target,
                    model_key=model_key,
                    excel_path="AI_Models_List.xlsx",
                    input_row=input_row_model,
                    memberName=info.get("member")
                )

                print("Model Trained:"+model_key)

                # Decode prediction
                pred_val = prediction[0] if hasattr(prediction, "__iter__") and not isinstance(prediction, str) else prediction
                if target in encoders:
                    final_prediction = encoders[target].inverse_transform([int(pred_val)])[0]
                else:
                    # Regression: inverse scale using raw mean/std
                    t_mean = raw_df[target].mean() if target in raw_df else 0
                    t_std = raw_df[target].std() if target in raw_df else 1
                    final_prediction = round(float(pred_val) * t_std + t_mean, 2)

            except Exception as e:
                print(f"Prediction error for {target} in {country_name}: {e}")
                final_prediction = "Error"

            prediction_entry["results"][target] = final_prediction

        mass_predictions.append(prediction_entry)

    return render_template("map.html", data=mass_predictions, targets=list(MEMBER_TARGET_MODELS.keys()))


# @app.route("/map")
# def map_route():
#     mass_predictions = []
#
#     raw_df = pd.read_csv(RAW_DATA_PATH)
#     cleaned_df = df.copy()
#
#     for country_name, coords in COUNTRY_COORDS.items():
#         input_row_template = cleaned_df.sample(1).copy()
#         for col in input_row_template.columns:
#             if col == 'Country':
#                 if country_name not in raw_df['Country'].values:
#                     # Assign a new integer ID to the new country
#                     new_country_id = cleaned_df['Country'].max() + 1  # or len(cleaned_df['Country'].values)
#                     input_row_template[col] = new_country_id
#
#                     # Update REVERSE_MAP for this new country
#                     if 'Country' not in REVERSE_MAP:
#                         REVERSE_MAP['Country'] = {}
#                     REVERSE_MAP['Country'][new_country_id] = country_name
#                 else:
#                     row_index = raw_df[raw_df['Country'] == country_name].index[0]
#                     input_row_template[col] = cleaned_df.loc[row_index, col]
#
#             elif col in ['Region', 'Crop_Type', 'Adaptation_Strategies']:
#                 # Fill with most frequent value
#                 input_row_template[col] = cleaned_df[col].mode()[0]
#
#             else:
#                 # Numeric columns: fill with mean
#                 if pd.api.types.is_numeric_dtype(cleaned_df[col]):
#                     input_row_template[col] = cleaned_df[col].mean()
#
#         # for cols in input_row_template.columns :
#         #     if cols =='Country' :
#         #         if country_name not in raw_df['Country'].values:
#         #             input_row_template[cols] = cleaned_df['Country'].values[len(cleaned_df['Country'].values)-1]+1
#         #             REVERSE_MAP[cols][input_row_template[cols]] = country_name
#         #         else :
#         #             row_index = raw_df[raw_df['Country'] == country_name].index[0]
#         #             input_row_template[cols] = cleaned_df.loc[row_index, cols]
#         #     elif cols in ['Region', 'Crop_Type', 'Adaptation_Strategies'] :
#         #         input_row_template[cols] = cleaned_df[cols].mode()[0]
#         #     else :
#         #         input_row_template[cols] = cleaned_df[cols].mean()
#
#
#         prediction_entry = {
#             "country": country_name,
#             "lat": float(coords[0]),
#             "lon": float(coords[1]),
#             "results": {}
#         }
#
#
#         #here i want to change the values of that row with the mean plz
#         for target, info in MEMBER_TARGET_MODELS.items():
#             models = info.get("models", [])
#             memberName = info.get("member")
#
#             if not models:
#                 prediction_entry["results"][target] = "None"
#                 continue
#
#             model_key = models[0]
#             input_row_model = input_row_template.drop(columns=[target], errors="ignore")
#
#             try:
#                 prediction, score, metric, _ = train_and_predict(
#                     df=df,
#                     target=target,
#                     model_key=model_key,
#                     excel_path="AI_Models_List.xlsx",
#                     input_row=input_row_model,
#                     memberName=memberName
#                 )
#                 print("Model Trained:"+model_key)
#
#
#             # 🔁 Decode prediction
#                 if target in REVERSE_MAP:
#                     final_prediction = REVERSE_MAP[target].get(prediction, "Unknown")
#                 else:
#                     mean = raw_df[target].mean()
#                     std = raw_df[target].std()
#                     final_prediction = round(prediction * std + mean, 2)
#
#             except Exception as e:
#                 print(f"Failed to train {model_key} for {country_name}: {e}")
#                 final_prediction = "None"
#
#             prediction_entry["results"][target] = final_prediction
#
#         mass_predictions.append(prediction_entry)
#
#     return render_template(
#         "map.html",
#         data=mass_predictions,
#         targets=list(MEMBER_TARGET_MODELS.keys())
#     )
@app.route("/predict_country")
def predict_country():
    country_name = request.args.get("country")
    if not country_name:
        return jsonify({"error": "No country provided"}), 400

    # Geocode country
    try:
        geolocator = Nominatim(user_agent="myapp")
        loc = geolocator.geocode(country_name)
        if not loc:
            return jsonify({"error": "Location not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Geocoding failed: {str(e)}"}), 500

    # Build input row (same logic as /map)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    training_data = df.copy()
    row_dict = {}

    if country_name in raw_df['Country'].values:
        country_row = raw_df[raw_df['Country'] == country_name].iloc[-1]
        for col in numeric_cols:
            row_dict[col] = country_row[col]
        for col in ['Region', 'Crop_Type', 'Adaptation_Strategies']:
            row_dict[col] = country_row[col]
        row_dict['Year'] = country_row['Year'] if 'Year' in country_row else 2024
    else:
        row_dict['Year'] = raw_df['Year'].max() if 'Year' in raw_df else 2024
        for col in numeric_cols:
            row_dict[col] = raw_df[col].mean()
        for col in ['Region', 'Crop_Type', 'Adaptation_Strategies']:
            row_dict[col] = raw_df[col].mode()[0]

    row_dict['Country'] = country_name
    input_row_raw = pd.DataFrame([row_dict])
    input_row_transformed = input_row_raw.copy()

    for col, le in encoders.items():
        if col in input_row_transformed.columns:
            val = str(input_row_transformed[col].iloc[0])
            input_row_transformed[col] = le.transform([val])[0] if val in le.classes_ else len(le.classes_)

    cols_to_scale = [c for c in numeric_cols if c in input_row_transformed.columns]
    if cols_to_scale:
        input_row_transformed[cols_to_scale] = scaler.transform(input_row_transformed[cols_to_scale])

    input_row_transformed = input_row_transformed[[col for col in training_data.columns if col in input_row_transformed.columns]]

    result_entry = {"country": country_name, "lat": loc.latitude, "lon": loc.longitude, "results": {}}

    for target, info in MEMBER_TARGET_MODELS.items():
        models = info.get("models", [])
        if not models:
            result_entry["results"][target] = None
            continue

        model_key = models[0]
        try:
            input_row_model = input_row_transformed.drop(columns=[target], errors="ignore")
            train_features = [col for col in training_data.columns if col != target]
            input_row_model = input_row_model[train_features]

            prediction, score, metric, _ = train_and_predict(
                df=training_data,
                target=target,
                model_key=model_key,
                excel_path="AI_Models_List.xlsx",
                input_row=input_row_model,
                memberName=info.get("member")
            )
            print("Model Trained:"+model_key)


            pred_val = prediction[0] if hasattr(prediction, "__iter__") and not isinstance(prediction, str) else prediction
            if target in encoders:
                final_prediction = encoders[target].inverse_transform([int(pred_val)])[0]
            else:
                t_mean = raw_df[target].mean() if target in raw_df else 0
                t_std = raw_df[target].std() if target in raw_df else 1
                final_prediction = round(float(pred_val) * t_std + t_mean, 2)

        except Exception as e:
            print(f"Prediction error for {target} in {country_name}: {e}")
            final_prediction = "Error"

        result_entry["results"][target] = final_prediction

    return jsonify(result_entry)




if __name__ == "__main__":
    app.run(debug=True)
