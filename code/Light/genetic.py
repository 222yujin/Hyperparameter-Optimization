import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from geneticalgorithm import geneticalgorithm as ga
from openpyxl import load_workbook
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 데이터 폴더 경로 지정
folder_path = 'C:/Users/yujin/Desktop/work/data/Preprocessed/Step2_Balanced/'
output_file = 'C:/Users/yujin/Desktop/work/result/lightgbm_genetic_algorithm_results.xlsx'

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

        # 유전 알고리즘을 위한 목적 함수 정의
        def fitness_function(params):
            n_estimators = int(params[0])
            max_depth = int(params[1])
            num_leaves = int(params[2])
            learning_rate = float(params[3])

            model = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return -f1_score(y_test, y_pred, average='binary')

        # 유전 알고리즘 설정
        varbound = np.array([[50, 300],  # n_estimators
                             [3, 15],    # max_depth
                             [20, 100],  # num_leaves
                             [0.01, 0.2] # learning_rate
                            ])
        algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1, 
                           'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 
                           'crossover_type': 'uniform', 'max_iteration_without_improv': None}

        model = ga(function=fitness_function, dimension=4, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
        model.run()

        # 최적의 하이퍼파라미터 추출
        best_params = model.output_dict['variable']
        n_estimators = int(best_params[0])
        max_depth = int(best_params[1])
        num_leaves = int(best_params[2])
        learning_rate = float(best_params[3])

        # 최적의 모델로 예측 수행
        best_lgb = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=42
        )
        best_lgb.fit(X_train, y_train)
        y_pred = best_lgb.predict(X_test)
        y_pred_proba = best_lgb.predict_proba(X_test)[:, 1]

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
            'Dataset': [dataset_name] * 4,
            'Parameter': ['n_estimators', 'max_depth', 'num_leaves', 'learning_rate'],
            'Best Value': [n_estimators, max_depth, num_leaves, learning_rate]
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
            # 파일이 존재하면 새로운 시트로 추가 저장
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                book = writer.book
                if f'{dataset_name} Performance Metrics' in book.sheetnames:
                    del book[f'{dataset_name} Performance Metrics']
                if f'{dataset_name} Best Hyperparameters' in book.sheetnames:
                    del book[f'{dataset_name} Best Hyperparameters']
                if f'{dataset_name} Classification Report' in book.sheetnames:
                    del book[f'{dataset_name} Classification Report']
                results_df.to_excel(writer, sheet_name=f'{dataset_name} Performance Metrics', index=False)
                params_df.to_excel(writer, sheet_name=f'{dataset_name} Best Hyperparameters', index=False)
                report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)

        print(f"엑셀 파일이 성공적으로 업데이트되었습니다: {dataset_name}")

