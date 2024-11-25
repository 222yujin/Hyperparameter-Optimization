import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터 예시
data = {
    'Model': ['Adaboost', 'Adaboost', 'Adaboost', 'Adaboost', 
              'CatBoost', 'CatBoost', 'CatBoost', 'CatBoost',
              'ExtraTree', 'ExtraTree', 'ExtraTree', 'ExtraTree', 
              'GradientBoosting', 'GradientBoosting', 'GradientBoosting', 'GradientBoosting',
              'LightGBM', 'LightGBM', 'LightGBM', 'LightGBM', 
              'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest', 
              'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'Optimization Method': ['Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm'],
    'Accuracy': [0.8904, 0.8984, 0.8887, 0.9117, 
                 0.914, 0.9129, 0.9177, 0.9339,
                 0.9225, 0.9221, 0.9194, 0.9196,
                 0.9249, 0.9257, 0.9278, 0.9372,
                 0.9357, 0.9293, 0.9273, 0.9239,
                 0.8908, 0.8931, 0.8858, 0.9256,
                 0.9151, 0.9143, 0.9095, 0.9386],
    'F1 Score': [0.8902, 0.8924, 0.8929, 0.9134, 
                 0.9161, 0.9147, 0.9183, 0.9274,
                 0.9163, 0.9173, 0.9156, 0.9158,
                 0.9194, 0.921, 0.9213, 0.9281,
                 0.9263, 0.9295, 0.9293, 0.9263,
                 0.878, 0.8935, 0.8758, 0.9197,
                 0.9104, 0.9091, 0.9067, 0.9379]
}

df = pd.DataFrame(data)

# Accuracy 그래프
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='Accuracy', hue='Optimization Method')
plt.ylim(0.9, 1.0)  # 세부 범위 강조
plt.title("Model Performance by Optimization Method (Accuracy)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.legend(title="Optimization Method")
plt.show()

# F1 Score 그래프
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='F1 Score', hue='Optimization Method')
plt.ylim(0.9, 1.0)  # 세부 범위 강조
plt.title("Model Performance by Optimization Method (F1 Score)")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.legend(title="Optimization Method")
plt.show()
