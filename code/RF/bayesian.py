import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import openpyxl
import os

# 데이터 로드
dataset_name = 'PC4'  # 데이터셋 이름 지정 (CM1, JM1, KC3 등)
file_path = f'C:\\Users\\user\\Desktop\\work\\data\\Preprocessed\\Step2_Balanced\\{dataset_name}_Clean_Balanced.csv'
data = pd.read_csv(file_path)

# 특성과 라벨 분리
X = data.drop('Defective', axis=1)
y = data['Defective']

# 데이터셋 분할 (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 초기화
rf = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 공간 설정 (베이지안 최적화용)
param_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Categorical([None, 10, 20, 30]),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'bootstrap': Categorical([True, False])
}

# 베이지안 최적화 설정
bayes_search = BayesSearchCV(
    estimator=rf,
    search_spaces=param_space,
    n_iter=30,  # 최적화 시도 횟수
    cv=5,  # 5겹 교차 검증
    n_jobs=-1,
    verbose=2,
    scoring='f1'
)

# 베이지안 최적화를 통한 하이퍼파라미터 최적화
bayes_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
best_params = bayes_search.best_params_
print("\n최적의 하이퍼파라미터 조합:")
print(best_params)

# 최적의 모델로 예측 수행
best_rf = bayes_search.best_estimator_
y_pred = best_rf.predict(X_test)

# AUC 값 계산을 위한 예측 확률값 사용
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]  # 양성 클래스(1)의 확률값 사용
auc = roc_auc_score(y_test, y_pred_proba)

# 성능 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# 분류 리포트 생성
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\n성능 지표:")
print(f"정확도: {accuracy:.4f}")
print(f"정밀도: {precision:.4f}")
print(f"재현율: {recall:.4f}")
print(f"F1 스코어: {f1:.4f}")
print(f"AUC: {auc:.4f}")  # AUC 값 출력

# 엑셀 파일에 결과 저장
output_file = 'C:/Users/user/Desktop/work/result/random_forest_bayesian_search_results.xlsx'

# 데이터프레임 생성 (AUC 값 추가)
results_df = pd.DataFrame({
    'Dataset': [dataset_name] * 5,
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Value': [accuracy, precision, recall, f1, auc]
})

params_df = pd.DataFrame({
    'Dataset': [dataset_name] * len(best_params),
    'Parameter': list(best_params.keys()),
    'Best Value': list(best_params.values())
})

# 엑셀 파일이 존재하는지 확인
if not os.path.exists(output_file):
    # 파일이 존재하지 않으면 새로운 파일 생성 및 데이터 저장
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        params_df.to_excel(writer, sheet_name='Best Hyperparameters', index=False)
        report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)
else:
    # 파일이 존재하면 데이터 추가
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        results_df.to_excel(writer, sheet_name='Performance Metrics', index=False, startrow=writer.sheets['Performance Metrics'].max_row)
        params_df.to_excel(writer, sheet_name='Best Hyperparameters', index=False, startrow=writer.sheets['Best Hyperparameters'].max_row)
        report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)

print("엑셀 파일이 업데이트되었습니다.")
