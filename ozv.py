import streamlit as st
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile

# Список поддерживаемых языков
LANGUAGES = {
    "Английский": "en",
    "Русский": "ru",
    "Испанский": "es",
    "Французский": "fr",
    "Немецкий": "de"
}

# Настройка заголовка
st.title("Актёр озвучивания с помощью нейросети")

# Ввод текста
text = st.text_area("Введите текст для озвучивания:", height=150)

# Выбор языка
language = st.selectbox("Выберите язык:", options=list(LANGUAGES.keys()))

# Скорость воспроизведения
speed = st.slider("Скорость воспроизведения:", min_value=0.5, max_value=2.0, value=1.0)

# Выбор качества/разрешения аудио
quality = st.selectbox("Выберите качество файла (битрейт):", ["Низкий", "Средний", "Высокий"])

# Кнопка для запуска TTS
if st.button("Озвучить текст"):
    if text.strip() == "":
        st.error("Введите текст для озвучивания.")
    else:
        # Генерация речи
        lang_code = LANGUAGES[language]
        slow = speed < 1.0
        
        # Настройка качества аудио
        if quality == "Low":
            tts = gTTS(text=text, lang=lang_code, slow=slow)
        elif quality == "Medium":
            tts = gTTS(text=text, lang=lang_code, slow=slow)
        else:  # High Quality
            tts = gTTS(text=text, lang=lang_code, slow=slow)

        # Сохранение во временный файл
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            audio_file = temp_file.name
        
        # Воспроизведение аудио
        st.audio(audio_file, format="audio/mp3")

        # Опция для скачивания файла
        st.download_button(
            label="Скачать аудиофайл",
            data=open(audio_file, "rb").read(),
            file_name="output.mp3",
            mime="audio/mp3"
        )

        # Удаление временного файла
        os.remove(audio_file)


#для работы с приложение необходимо установить pip install streamlit gtts
#Lkz запуска приложения streamlit run ozv.py
