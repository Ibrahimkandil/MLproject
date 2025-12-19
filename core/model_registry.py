# Each member owns ONE target with 3 models

MEMBER_TARGET_MODELS = {
    "Economic_Impact_Million_USD": {
        "member": "Ibrahim Kandil",
        "task": "regression",
        "models": [
            "LINEAR_REGRESSION",
            "MLP_REG",
            "RANDOM_FOREST_REG"
        ]
    },

    "Crop_Type": {
        "member": "Mayssem Ben Hammouda",
        "task": "classification",
        "models": [
            "LOGISTIC_REGRESSION",
            "RANDOM_FOREST_CLS",
            "DECISION_TREE_CLS"
        ]
    },

    "Adaptation_Strategies": {
        "member": "Rim Khiari",
        "task": "classification",
        "models": [
            "XGBOOST_CLS",
            "GRADIENT_BOOSTING_CLS",
            "SVM_CLS"
        ]
    },

    "Extreme_Weather_Events": {
        "member": "Oumaima Issaoui",
        "task": "regression",
        "models": [
            "RIDGE",
            "LASSO",
            "DECISION_TREE_REG"
        ]
    },

    "CO2_Emissions_MT": {
        "member": "Nouhen Chaouch",
        "task": "regression",
        "models": [
            "ELASTICNET",
            "KNN_REG",
            "EXTRA_TREES_REG"
        ]
    }
}
