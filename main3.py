# seaborn 불러오기
import seaborn as sns
#matplotlib 불러오기
from matplotlib import pyplot as plt

#plt.subplots함수로 "도화지(figure)" 및 "축(axis)" 그리기
fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.flatten()

# 그래프 제목 정하기
fig.suptitle('time count visualization')

#sns.countplot함수를 이용하여 그래프 그리기
sns.countplot(train["year"],ax = axes[0])
sns.countplot(train["month"],ax = axes[1])
sns.countplot(train["day"],ax = axes[2])
sns.countplot(train["hour"],ax = axes[3])
sns.countplot(train["minute"],ax = axes[4])
sns.countplot(train["second"],ax = axes[5])
fig.show()
fig.show()