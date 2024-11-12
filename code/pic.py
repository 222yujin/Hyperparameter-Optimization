import pandas as pd
import matplotlib.pyplot as plt

# 전처리 전후 데이터셋 불러오기
data_before = pd.read_csv("C:\\Users\\user\\Desktop\work\data\Original\CM1.csv")
data_after = pd.read_csv("C:\\Users\\user\\Desktop\work\data\Preprocessed\Step2_Balanced\CM1_Clean_Balanced.csv")

# 기초 통계량 비교
print("Before Processing:\n", data_before.describe())
print("After Processing:\n", data_after.describe())

# 결측치 비교
print("Missing Values Before:\n", data_before.isnull().sum())
print("Missing Values After:\n", data_after.isnull().sum())

# 히스토그램 비교 (예시)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data_before['Defective'], bins=20, alpha=0.7, label='Before')
plt.title('Before Processing')
plt.subplot(1, 2, 2)
plt.hist(data_after['Defective'], bins=20, alpha=0.7, label='After')
plt.title('After Processing')
plt.show()
