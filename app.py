import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import ConfusionMatrixDisplay

# Импорт кастомных модулей финального проекта
from data_manager import DataManager
from model_engine import ModelEngine

# Настройка страницы
st.set_page_config(
    page_title="Modular AI Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Хранение состояния сессии
if "model_engine" not in st.session_state:
    st.session_state.model_engine = None
    st.session_state.data_manager = DataManager()
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.metrics = None

# --- ШАПКА ПРИЛОЖЕНИЯ ---
st.title("🧠 Модульный AI-симулятор: Проектирование и обучение нейросетей")
st.markdown(
    "Данный симулятор демонстрирует принципы проектирования искусственного интеллекта. "
    "Проект разделен на три модуля: **генератор данных**, **математическое ядро классификации** и **веб-интерфейс**."
)

# --- БОКОВАЯ ПАНЕЛЬ: Параметры конфигурации ---
st.sidebar.title("🛠️ Панель конфигурации")
st.sidebar.markdown("---")

# 1. Выбор движка ИИ
st.sidebar.subheader("🔌 Движок вычислений")
engine_option = st.sidebar.selectbox(
    "Вычислительное ядро:",
    ["Scikit-learn (MLP)", "TensorFlow/Keras"] if ModelEngine.is_keras_supported() else ["Scikit-learn (MLP) (Keras недоступен)"]
)
engine_type = "TensorFlow/Keras" if "TensorFlow" in engine_option else "Scikit-learn (MLP)"

# 2. Выбор датасета
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Параметры выборки")
dataset_type = st.sidebar.selectbox(
    "Набор данных (распределение):",
    ["Полумесяцы (make_moons)", "Круги (make_circles)", "Блоки (make_blobs)"]
)
n_samples = st.sidebar.slider("Общее число точек:", 100, 1000, 400, step=50)
noise_val = st.sidebar.slider("Уровень шума (дисперсия):", 0.05, 0.45, 0.15, step=0.05)
test_split = st.sidebar.slider("Размер тестовой выборки:", 0.1, 0.4, 0.2, step=0.05)

# 3. Конфигуратор слоев
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Архитектура сети")
neurons_l1 = st.sidebar.slider("Нейронов в 1-м скрытом слое:", 4, 32, 16, step=4)
neurons_l2 = st.sidebar.slider("Нейронов во 2-м скрытом слое:", 0, 16, 8, step=2)
neurons_l3 = st.sidebar.slider("Нейронов в 3-м скрытом слое:", 0, 8, 0, step=2)

activation_fn = st.sidebar.selectbox(
    "Функция активации нейронов:",
    ["relu", "tanh", "logistic", "identity"]
)

# 4. Скорость обучения
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Режим обучения")
epochs_val = st.sidebar.slider("Количество эпох (итераций):", 5, 100, 30, step=5)
lr_val = st.sidebar.selectbox("Начальный шаг обучения (Learning Rate):", [0.001, 0.01, 0.1, 0.5], index=1)

# Формирование кортежа скрытых слоев
hidden_layers_list = [neurons_l1]
if neurons_l2 > 0:
    hidden_layers_list.append(neurons_l2)
if neurons_l3 > 0:
    hidden_layers_list.append(neurons_l3)
hidden_layers = tuple(hidden_layers_list)

# --- Генерация данных при изменении боковой панели ---
X, y = DataManager.generate_dataset(dataset_type, n_samples, noise_val)
X_train, X_test, y_train, y_test = DataManager.prepare_data(X, y, test_split)

# Вкладки приложения
tab1, tab2, tab3 = st.tabs([
    "📊 Модуль 1: Анализ данных (Pandas)", 
    "📈 Модуль 2: Визуализатор обучения (Matplotlib)", 
    "🏆 Модуль 3: Оценка модели и ИИ-Тестер"
])

# ==================== ВКЛАДКА 1: МОДУЛЬ ДАННЫХ ====================
with tab1:
    st.header("📊 Модуль обработки данных (data_manager.py)")
    st.markdown("В этом разделе студенты анализируют сгенерированные точки с помощью Pandas.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Табличное представление (DataFrame)")
        # Создание датафрейма для анализа
        df = pd.DataFrame(X, columns=["Признак_X1", "Признак_X2"])
        df["target"] = y
        st.dataframe(df, use_container_width=True, height=350)
        
    with col2:
        st.subheader("Сводный анализ и статистика")
        st.write("Сводная статистика по числовым признакам:")
        st.write(df.describe())
        
        # Распределение классов
        class_summary = DataManager.get_metrics_summary(df)
        st.write("Распределение классов в наборе:")
        for cls, count in class_summary.items():
            st.info(f"Класс {cls}: **{count} точек** ({count/len(df)*100:.1f}%)")

# ==================== ВКЛАДКА 2: ВИЗУАЛИЗАТОР ОБУЧЕНИЯ ====================
with tab2:
    st.header("📈 Модуль живого обучения и визуализации (model_engine.py)")
    st.markdown("Здесь визуализируется пошаговый процесс оптимизации весов нейросети.")
    
    if st.button("🚀 Начать обучение нейросети"):
        # Создание движка модели
        engine = ModelEngine(
            engine_type=engine_type,
            hidden_layers=hidden_layers,
            activation=activation_fn,
            learning_rate=lr_val
        )
        engine.initialize_model(input_dim=2)
        
        # Резервирование мест под живые графики
        plot_loss = st.empty()
        plot_acc = st.empty()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history_loss = []
        history_acc = []
        
        classes = np.unique(y_train)
        
        # Пошаговое обучение
        for epoch in range(epochs_val):
            loss, acc = engine.train_epoch(X_train, y_train, classes=classes)
            
            history_loss.append(loss)
            history_acc.append(acc)
            
            # Живая отрисовка Loss
            fig_loss, ax_loss = plt.subplots(figsize=(8, 2.5))
            ax_loss.set_facecolor('#0f131f')
            fig_loss.patch.set_facecolor('#080b11')
            ax_loss.plot(history_loss, color="#e74c3c", label="Loss (Ошибка)", marker="o", markersize=2)
            ax_loss.set_title(f"Кривая ошибки (Эпоха {epoch+1}/{epochs_val})", color="#f1f5f9")
            ax_loss.tick_params(colors="#94a3b8")
            ax_loss.legend(facecolor='#0f131f', edgecolor='#1e293b', labelcolor='#f1f5f9')
            plt.tight_layout()
            plot_loss.pyplot(fig_loss)
            plt.close(fig_loss)
            
            # Живая отрисовка Accuracy
            fig_acc, ax_acc = plt.subplots(figsize=(8, 2.5))
            ax_acc.set_facecolor('#0f131f')
            fig_acc.patch.set_facecolor('#080b11')
            ax_acc.plot(history_acc, color="#2ecc71", label="Accuracy (Точность)", marker="x", markersize=2)
            ax_acc.set_title(f"Кривая точности (Эпоха {epoch+1}/{epochs_val})", color="#f1f5f9")
            ax_acc.tick_params(colors="#94a3b8")
            ax_acc.legend(facecolor='#0f131f', edgecolor='#1e293b', labelcolor='#f1f5f9')
            plt.tight_layout()
            plot_acc.pyplot(fig_acc)
            plt.close(fig_acc)
            
            # Прогресс
            progress_bar.progress(int(((epoch+1)/epochs_val)*100))
            status_text.text(f"Эпоха {epoch+1} завершена. Текущий Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            
            time.sleep(0.05)
            
        # Тестирование и оценка результатов
        metrics = engine.evaluate(X_test, y_test)
        
        # Сохранение состояния
        st.session_state.model_engine = engine
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.metrics = metrics
        
        st.success("🎉 Обучение полностью завершено. Перейдите на Вкладку 3 для тестирования модели.")
        st.balloons()

# ==================== ВКЛАДКА 3: ОЦЕНКА И ТЕСТИРОВАНИЕ ====================
with tab3:
    st.header("🏆 Модуль оценки качества работы искусственного интеллекта")
    
    if st.session_state.model_engine is not None and st.session_state.model_engine.is_trained or st.session_state.metrics is not None:
        engine = st.session_state.model_engine
        metrics = st.session_state.metrics
        
        # Метрики качества на тесте
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Тестовая точность (Accuracy)", f"{metrics['accuracy']*100:.2f}%")
        col_m2.metric("Точность прогноза (Precision)", f"{metrics['precision']*100:.2f}%")
        col_m3.metric("Полнота отбора (Recall)", f"{metrics['recall']*100:.2f}%")
        
        st.markdown("---")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.subheader("🗺️ Карта классификации пространства ИИ")
            
            # Получение тепловой карты и границы решения
            xx, yy, Z = engine.get_decision_boundary(X, resolution=60)
            
            fig_map, ax_map = plt.subplots(figsize=(6, 4.5))
            ax_map.set_facecolor('#0f131f')
            fig_map.patch.set_facecolor('#080b11')
            
            # Заливка пространства уверенности ИИ
            ax_map.contourf(xx, yy, Z, alpha=0.75, cmap="RdYlGn", levels=15)
            # Отрисовка исходных точек выборки
            ax_map.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", cmap="RdYlGn_r", s=30, label="Обучение")
            ax_map.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="w", cmap="RdYlGn_r", s=40, marker="D", label="Тест")
            
            ax_map.set_xlabel("X1", color="#94a3b8")
            ax_map.set_ylabel("X2", color="#94a3b8")
            ax_map.tick_params(colors="#94a3b8")
            ax_map.legend()
            plt.tight_layout()
            st.pyplot(fig_map)
            plt.close(fig_map)
            
        with col_res2:
            st.subheader("📊 Матрица ошибок (Confusion Matrix)")
            st.markdown("Матрица ошибок показывает количество верных и ложных предсказаний для каждого класса на тесте.")
            
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
            fig_cm.patch.set_facecolor('#080b11')
            
            disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'], display_labels=["Класс 0", "Класс 1"])
            disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.title.set_color('#f1f5f9')
            ax_cm.xaxis.label.set_color('#94a3b8')
            ax_cm.yaxis.label.set_color('#94a3b8')
            ax_cm.tick_params(colors="#94a3b8")
            
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            
        st.markdown("---")
        
        # Блок интерактивного тестера для родителей
        st.subheader("🔮 Демонстрация ИИ для родителей: Проверь собственную точку")
        st.markdown("Введите любые значения признаков X1 и X2, чтобы нейросеть мгновенно классифицировала их.")
        
        col_p1, col_p2 = st.columns(2)
        p_x1 = col_p1.number_input("Координата по оси X1 (признак 1):", value=0.0, step=0.1)
        p_x2 = col_p2.number_input("Координата по оси X2 (признак 2):", value=0.0, step=0.1)
        
        if st.button("🔍 Запустить мгновенное предсказание"):
            test_point = np.array([[p_x1, p_x2]])
            pred_class, conf_val = engine.predict_point(test_point)
            
            st.markdown("#### Вердикт нейронной сети:")
            if pred_class == 1:
                st.success(f"🟢 Точка отнесена к Классу 1 (Зеленый)")
                st.write(f"Уверенность модели: **{conf_val * 100:.2f}%**")
                st.progress(conf_val)
            else:
                st.error(f"🔴 Точка отнесена к Классу 0 (Красный)")
                st.write(f"Уверенность модели: **{conf_val * 100:.2f}%**")
                st.progress(conf_val)
                
        # Блок экспорта модели
        st.markdown("---")
        st.subheader("💾 Экспорт обученной модели")
        st.markdown("Студенты могут выгрузить веса обученной модели для использования в реальном приложении.")
        
        if st.button("💾 Сохранить и скачать модель"):
            model_name = f"model_{engine_type.lower().replace(' ', '_').replace('/', '_')}"
            engine.save_model(model_name)
            ext = ".h5" if engine_type == "TensorFlow/Keras" else ".pkl"
            
            if os.path.exists(model_name + ext):
                with open(model_name + ext, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Скачать файл {model_name}{ext}",
                        data=f,
                        file_name=model_name + ext,
                        mime="application/octet-stream"
                    )
                    st.success("Файл модели успешно подготовлен к выгрузке.")
            else:
                st.error("Ошибка сохранения модели. Попробуйте еще раз.")
    else:
        st.info("Модель еще не обучена. Перейдите на Вкладку 2 и нажмите кнопку обучения.")
