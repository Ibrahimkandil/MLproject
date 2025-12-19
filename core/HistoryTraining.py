History = []

def initialize() :
    History = []
def CopyAtrainingHistoryFrom(model_name, model_obj, member, target, score, prediction, learning_type) :
    from datetime import datetime
    training_history = {
        "ModelName": model_name,
        "Model": str(model_obj),  # string representation
        "Member": member,
        "Target": target,
        "Score": score,
        "Prediction": prediction,
        "LearningType": learning_type,
        "CreatedTime": datetime.now()
    }
    History.append(training_history)
def GetHistory() :
    return History
