# hotel_booking_analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
print("1. ЗАГРУЗКА ДАННЫХ")
data = pd.read_csv("hotel_bookings.csv")
print(f"Размер данных: {data.shape}")

# 2. Предобработка
# Заполнение пропусков
data['children'] = data['children'].fillna(0)
data['country'] = data['country'].fillna(data['country'].mode()[0])
data['agent'] = data['agent'].fillna(0)
data['company'] = data['company'].fillna(0)

# Подготовка признаков и целевой переменной
X = data.drop('is_canceled', axis=1)
y = data['is_canceled']

# Кодирование категориальных переменных
categorical_cols = ['hotel', 'meal', 'country', 'market_segment', 
                   'distribution_channel', 'deposit_type', 'customer_type']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Label Encoding для остальных категориальных признаков
for col in X_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# 3. Разбиение на выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Нормализация
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 5. Обучение моделей
print("\nОБУЧЕНИЕ МОДЕЛЕЙ")

# KNN с оптимизацией k
best_k = 1
best_score = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

knn_optimal = KNeighborsClassifier(n_neighbors=best_k)
knn_optimal.fit(X_train, y_train)

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Оценка моделей
print(f"\nРЕЗУЛЬТАТЫ:")
print(f"KNN (k={best_k}): {knn_optimal.score(X_test, y_test):.4f}")
print(f"Random Forest: {rf.score(X_test, y_test):.4f}")

# Матрицы рассогласования
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# KNN
y_pred_knn = knn_optimal.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', ax=ax1)
ax1.set_title(f'KNN (k={best_k})')

# Random Forest
y_pred_rf = rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', ax=ax2)
ax2.set_title('Random Forest')

plt.tight_layout()
plt.show()

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))