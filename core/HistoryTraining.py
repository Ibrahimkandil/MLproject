History = []

def initialize() :
    History = []
def CopyAtrainingHistoryFrom(model_name, model_obj, member, target, score, prediction, learning_type) :
    from datetime import datetime
    training_history = {
        "ModelName": model_name,
        "ModelObj": model_obj,      # Store the ACTUAL object here (don't use str())
        "Member": member,
        "Target": target,
        "Score": score,
        "Prediction": prediction,   # This is your original training prediction
        "LearningType": learning_type.lower(),
        "CreatedTime": datetime.now()
    }
    History.append(training_history)
def GetHistory() :
    return History
