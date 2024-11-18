import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.formula.api import ols

# 예시 데이터프레임 생성
data = {
    'Model': ['Adaboost', 'Adaboost', 'Adaboost', 'Adaboost', 
              'CatBoost', 'CatBoost', 'CatBoost', 'CatBoost', 
              'ExtraTree', 'ExtraTree', 'ExtraTree', 'ExtraTree', 
              'GradientBoosting', 'GradientBoosting', 'GradientBoosting', 'GradientBoosting', 
              'LightGBM', 'LightGBM', 'LightGBM', 'LightGBM', 
              'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest', 
              'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'Optimization_Method': ['Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm',
                            'Grid Search', 'Random Search', 'Bayesian Search', 'Genetic Algorithm'],
    'F1_Score': [0.8902, 0.8924, 0.8929, 0.9134, 
                 0.9161, 0.9147, 0.9183, 0.9274, 
                 0.9163, 0.9173, 0.9156, 0.9158, 
                 0.9194, 0.9210, 0.9213, 0.9281, 
                 0.9263, 0.9295, 0.9293, 0.9263, 
                 0.8780, 0.8935, 0.8758, 0.9197, 
                 0.9104, 0.9091, 0.9067, 0.9379]
}

df = pd.DataFrame(data)

# 문자열이나 기타 비숫자 데이터가 있을 수 있으므로 숫자형으로 변환
df['F1_Score'] = pd.to_numeric(df['F1_Score'], errors='coerce')

# NaN, inf 값 점검 및 제거
if df.isnull().values.any():
    print("Data contains NaN values.")
if np.isinf(df.select_dtypes(include=[np.number]).values).any():
    print("Data contains inf or -inf values.")

# NaN 및 inf 값 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 다시 점검 (확인용)
print("Cleaned DataFrame:")
print(df)

# 이원 분산 분석 수행
model = ols('F1_Score ~ C(Model) + C(Optimization_Method) + C(Model):C(Optimization_Method)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
