import matplotlib.pyplot as plt
import pandas as pd

# 데이터 정의
data = {
    'Optimization Method': ['Bayesian Search', 'Genetic Algorithm', 'Grid Search', 'Random Search'],
    'Average Accuracy': [0.92, 0.94, 0.93, 0.91]
}

df = pd.DataFrame(data)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.bar(df['Optimization Method'], df['Average Accuracy'], color=['skyblue', 'salmon', 'lightgreen', 'coral'])
plt.ylim(0.88, 0.95)  # 축소된 범위
plt.title('Average Accuracy Across Models by Optimization Method', fontsize=14)
plt.xlabel('Optimization Method', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
