import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from geneticalgorithm import geneticalgorithm as ga
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from openpyxl import load_workbook
import warnings
import matplotlib
matplotlib.use('Agg')  # 이미지 백엔드 설정

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 데이터 폴더 경로 지정
folder_path = 'C:/Users/yujin/Desktop/work/data/Preprocessed/Step2_Balanced/'
output_file = 'C:/Users/yujin/Desktop/work/result/xgboost_genetic_algorithm_results.xlsx'

# 유전 알고리즘의 파라미터 범위 설정
def fitness_function(solution, solution_idx):
    params = {
        'learning_rate': solution[0],
        'n_estimators': int(solution[1]),
        'max_depth': int(solution[2]),
        'min_child_weight': solution[3],
        'subsample': solution[4],
        'colsample_bytree': solution[5]
    }
    try:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='binary')
        return -f1  # 유전 알고리즘은 최소화를 목표로 하므로 음수 처리
    except Exception as e:
        return float('inf')  # 오류가 발생한 경우, 높은 값을 반환하여 해당 솔루션이 선택되지 않도록 함

varbound = np.array([
    [0.01, 0.3],  # learning_rate
    [50, 300],    # n_estimators
    [3, 10],      # max_depth
    [1, 10],      # min_child_weight
    [0.8, 1.0],   # subsample
    [0.8, 1.0]    # colsample_bytree
])

algorithm_param = {'max_num_iteration': 50, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}

# 폴더 내의 모든 CSV 파일에 대해 반복 수행
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        dataset_name = filename.split('_')[0]  # 파일명에서 데이터셋 이름 추출 (예: CM1, JM1 등)
        file_path = os.path.join(folder_path, filename)

        # 데이터 로드
        data = pd.read_csv(file_path)

        # 특성과 라벨 분리
        X = data.drop('Defective', axis=1)
        y = data['Defective']

        # 데이터셋 분할 (train/test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 유전 알고리즘 초기화 및 실행
        model = ga(function=fitness_function, dimension=6, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param, function_timeout=30)
        model.run()

        # 최적의 하이퍼파라미터 추출
        best_solution = model.output_dict['variable']
        best_params = {
            'learning_rate': best_solution[0],
            'n_estimators': int(best_solution[1]),
            'max_depth': int(best_solution[2]),
            'min_child_weight': best_solution[3],
            'subsample': best_solution[4],
            'colsample_bytree': best_solution[5]
        }
        print(f"\n[{dataset_name}] 최적의 하이퍼파라미터 조합:")
        print(best_params)

        # 최적의 모델로 예측 수행
        best_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **best_params)
        best_xgb.fit(X_train, y_train)
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

        # 성능 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_pred_proba)

        # 분류 리포트 생성
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print(f"\n[{dataset_name}] 성능 지표:")
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1 스코어: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # 최적 하이퍼파라미터 저장
        params_df = pd.DataFrame({
            'Dataset': [dataset_name] * len(best_params),
            'Parameter': list(best_params.keys()),
            'Best Value': list(best_params.values())
        })

        # 엑셀 파일에 결과 저장
        # 데이터프레임 생성 (AUC 값 추가)
        results_df = pd.DataFrame({
            'Dataset': [dataset_name] * 5,
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
            'Value': [accuracy, precision, recall, f1, auc]
        })

        # 엑셀 파일이 존재하는지 확인
        if not os.path.exists(output_file):
            # 파일이 존재하지 않으면 새로운 파일 생성 및 데이터 저장
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=f'{dataset_name} Performance Metrics', index=False)
                params_df.to_excel(writer, sheet_name=f'{dataset_name} Best Hyperparameters', index=False)
                report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)
        else:
            # 파일이 존재하면 기존 파일 불러오기 및 시트 추가
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                results_df.to_excel(writer, sheet_name=f'{dataset_name} Performance Metrics', index=False)
                params_df.to_excel(writer, sheet_name=f'{dataset_name} Best Hyperparameters', index=False)
                report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)

        print(f"엑셀 파일이 성공적으로 업데이트되었습니다: {dataset_name}")
