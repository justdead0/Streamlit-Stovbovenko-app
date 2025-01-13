import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment

# Функция для перевода аудио в текст
def audio_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='ru-RU') # Пример для русского языка. Можете выбрать нужный язык.
    return text

# Тело веб-приложения
def main():
    st.title("Преобразование аудио в текст")
    audio_file = st.file_uploader("Загрузите аудиофайл", type=["mp3", "wav"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav') # Отображение аудиофайла

    if st.button("Получить текст из аудио"):
        if audio_file is not None:
            with open("temp_audio.wav", "wb") as file:
                file.write(audio_file.getbuffer())
            text = audio_to_text("temp_audio.wav")
            st.write("Текст:")
            st.write(text)

# Запуск приложения
if __name__ == '__main__':
    main()
