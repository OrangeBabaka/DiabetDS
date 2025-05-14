# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
file_path = "diabetes_prediction_dataset.csv"  # Укажите путь к вашему CSV-файлу
data = pd.read_csv(file_path)

# Шаг 2: Предварительная обработка данных

# Заменяем "No Info" на NaN для упрощения обработки
data.replace("No Info", None, inplace=True)

# Кодируем категориальные переменные (например, gender и smoking_history)
label_encoders = {}
for column in ['gender', 'smoking_history']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].fillna("Unknown"))

# Заполняем пропущенные значения в числовых столбцах средними значениями
for column in ['bmi', 'HbA1c_level', 'blood_glucose_level']:
    data[column].fillna(data[column].mean(), inplace=True)

# Шаг 3: Разделение на обучающую и тестовую выборки
X = data.drop(['diabetes'], axis=1)  # Все признаки, кроме целевой переменной
y = data['diabetes']  # Целевая переменная

# Масштабируем числовые данные
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 4: Обучение модели логистической регрессии (метод максимального правдоподобия - MLE)
model = LogisticRegression(solver='lbfgs', max_iter=1000)  # lbfgs использует MLE
model.fit(X_train, y_train)

# Шаг 5: Предсказание на тестовых данных
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Вероятности принадлежности к классу 1

# Шаг 6: Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
mse = mean_squared_error(y_test, y_proba)

print("\n=== Оценка качества модели ===")
print(f"Accuracy (Точность): {accuracy:.4f}")
print(f"ROC-AUC (Площадь под кривой): {roc_auc:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title("Матрица ошибок")
plt.xlabel("Предсказано")
plt.ylabel("Истинное")
plt.show()

# Шаг 7: Визуализация распределения предсказанных вероятностей
plt.figure(figsize=(8, 6))
sns.histplot(y_proba, kde=True, bins=20, color='blue', alpha=0.7)
plt.title("Распределение предсказанных вероятностей")
plt.xlabel("Предсказанная вероятность (0 = Низкий риск, 1 = Высокий риск)")
plt.ylabel("Количество")
plt.show()

# Шаг 8: Вывод итогов в DataFrame
results = pd.DataFrame({
    "Истинное значение": y_test.values,
    "Предсказанная вероятность": y_proba,
    "Класс (Предсказанный)": y_pred
})

# Добавляем интерпретацию вероятностей
def interpret_risk(probability):
    if probability < 0.3:
        return "Низкая предрасположенность"
    elif probability < 0.7:
        return "Средняя предрасположенность"
    else:
        return "Высокая предрасположенность"

results['Интерпретация риска'] = results['Предсказанная вероятность'].apply(interpret_risk)

print("\n=== Пример предсказаний ===")
print(results.head(10))  # Показываем первые 10 строк