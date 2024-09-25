import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from tpot import TPOTClassifier
import openpyxl
import os

# 데이터 로드
dataset_name = 'PC4'  # 데이터셋 이름 지정 CM1 JM1 KC1 KC3 MC1 MC2 MW1 PC1 PC2 PC3 PC4
file_path = f'C:\\Users\\user\\Desktop\\work\\data\\Preprocessed\\Step2_Balanced\\{dataset_name}_Clean_Balanced.csv'
data = pd.read_csv(file_path)

# 특성과 라벨 분리
X = data.drop('Defective', axis=1)
y = data['Defective']

# 데이터셋 분할 (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 유전 알고리즘을 이용한 TPOT 설정
tpot = TPOTClassifier(
    generations=5,  # 진화 세대 수
    population_size=50,  # 각 세대에서 탐색할 개체 수
    verbosity=2,
    random_state=42,
    scoring='f1',  # F1 스코어를 사용하여 성능 평가
    cv=5  # 5겹 교차 검증
)

# 유전 알고리즘을 사용한 모델 학습 및 최적화
tpot.fit(X_train, y_train)

# 최적화된 모델로 예측 수행
y_pred = tpot.predict(X_test)

# AUC 값 계산을 위한 예측 확률값 사용
y_pred_proba = tpot.predict_proba(X_test)[:, 1]  # 양성 클래스(1)의 확률값 사용
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
output_file = 'C:/Users/user/Desktop/work/result/tpot_genetic_algorithm_results.xlsx'

# 데이터프레임 생성 (AUC 값 추가)
results_df = pd.DataFrame({
    'Dataset': [dataset_name] * 5,
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Value': [accuracy, precision, recall, f1, auc]
})

# 최적화된 모델 정보 추출
model_code = tpot.export()
params_df = pd.DataFrame({
    'Dataset': [dataset_name],
    'Optimal Model': [model_code]  # 최적화된 모델 코드 저장
})

# 엑셀 파일이 존재하는지 확인
if not os.path.exists(output_file):
    # 파일이 존재하지 않으면 새로운 파일 생성 및 데이터 저장
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        params_df.to_excel(writer, sheet_name='Best Model', index=False)
        report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)
else:
    # 파일이 존재하면 데이터 추가
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        results_df.to_excel(writer, sheet_name='Performance Metrics', index=False, startrow=writer.sheets['Performance Metrics'].max_row)
        params_df.to_excel(writer, sheet_name='Best Model', index=False, startrow=writer.sheets['Best Model'].max_row)
        report_df.to_excel(writer, sheet_name=f'{dataset_name} Classification Report', index=True)

print("엑셀 파일이 업데이트되었습니다.")
