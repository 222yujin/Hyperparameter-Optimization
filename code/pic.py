import pandas as pd

# 데이터 생성
data = {
    "Model": ["Adaboost", "", "", "", "CatBoost", "", "", "", "ExtraTree", "", "", "",
              "GradientBoosting", "", "", "", "LightGBM", "", "", "", "RandomForest", "", "", "", "XGBoost", "", "", ""],
    "Optimization Method": ["Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm",
                            "Grid Search", "Random Search", "Bayesian Search", "Genetic Algorithm"],
    "Accuracy": [0.8904, 0.8984, 0.8887, 0.9117,
                 0.9140, 0.9129, 0.9177, 0.9339,
                 0.9225, 0.9221, 0.9194, 0.9196,
                 0.9249, 0.9257, 0.9278, 0.9372,
                 0.9357, 0.9293, 0.9273, 0.9239,
                 0.8908, 0.8931, 0.8858, 0.9256,
                 0.9151, 0.9143, 0.9095, 0.9386],
    "Precision": [0.8874, 0.8842, 0.8919, 0.9117,
                  0.9049, 0.9041, 0.9034, 0.9167,
                  0.9067, 0.9064, 0.9048, 0.9041,
                  0.9068, 0.9086, 0.9069, 0.9089,
                  0.9068, 0.9082, 0.9051, 0.9176,
                  0.8879, 0.8983, 0.8797, 0.9223,
                  0.8957, 0.8935, 0.8828, 0.9283],
    "Recall": [0.9137, 0.9180, 0.9075, 0.9267,
               0.9547, 0.9542, 0.9588, 0.9522,
               0.9382, 0.9390, 0.9325, 0.9329,
               0.9362, 0.9388, 0.9387, 0.9492,
               0.9529, 0.9568, 0.9568, 0.9614,
               0.8824, 0.8911, 0.8831, 0.9196,
               0.9349, 0.9318, 0.9341, 0.9466],
    "F1 Score": [0.8902, 0.8924, 0.8929, 0.9134,
                 0.9161, 0.9147, 0.9183, 0.9274,
                 0.9163, 0.9173, 0.9156, 0.9158,
                 0.9194, 0.9210, 0.9213, 0.9281,
                 0.9263, 0.9295, 0.9293, 0.9263,
                 0.8780, 0.8935, 0.8758, 0.9197,
                 0.9104, 0.9091, 0.9067, 0.9379],
    "AUC": [0.9361, 0.9448, 0.9511, 0.9464,
            0.9694, 0.9718, 0.9726, 0.9763,
            0.9685, 0.9695, 0.9693, 0.9702,
            0.9757, 0.9761, 0.9759, 0.9813,
            0.9809, 0.9812, 0.9814, 0.9762,
            0.9319, 0.9351, 0.9371, 0.9407,
            0.9654, 0.9647, 0.9649, 0.9571]
}

df = pd.DataFrame(data)

# Display the DataFrame
import ace_tools as tools; 
tools.display_dataframe_to_user(name="Model Optimization Performance Data", dataframe=df)
