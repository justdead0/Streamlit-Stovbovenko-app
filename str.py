import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Загрузка модели MobileNetV2
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Функция для предобработки изображения
def preprocess_image(image):
    img = image.resize((224, 224))  # Изменяем размер изображения
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Преобразуем изображение в массив
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Предобработка
    return img_array

# Функция для классификации изображения
def classify_image(image):
    img_array = preprocess_image(image) 
    predictions = model.predict(img_array)  # Получаем предсказания
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]  # Декодируем результаты
    return decoded_predictions

# Заголовок приложения
st.title("Классификация изображений с помощью нейросети")

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Выберите изображение для классификации", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    # Классификация изображения
    if st.button("Классифицировать"):
        with st.spinner("Классификация..."):
            predictions = classify_image(image)  # Получаем предсказания
            st.subheader("Предсказания:")
            for pred in predictions:
                st.write(f"**{pred[1]}**: {pred[2] * 100:.2f}%")  # Название и вероятность


# для работы данного приложение понадобится pip install streamlit tensorflow pillow
# для запуска приложения streamlit run str.py
