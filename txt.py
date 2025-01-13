import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# Инициализация модели и токенизатора
tokenizer, model = load_model()

# Настройка интерфейса Streamlit
st.title("Генерация  текста на основе слова")
st.write("Введите слово, и модель сгенерирует осмысленный текст на его основе.")

# Ввод от пользователя
input_word = st.text_input("Введите слово:", "")

# Кнопка для генерации текста
if st.button("Сгенерировать текст"):
    if input_word.strip():
        # Генерация текста
        input_ids = tokenizer.encode(input_word, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=50,  # Максимальная длина генерируемого текста
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Отображение результата
        st.subheader("Сгенерированный текст:")
        st.write(generated_text)
    else:
        st.warning("Пожалуйста, введите слово для генерации текста.")


# для работы приложения необходимо pip install streamlit transformers torch
 # для запуска приложения streamlit run txt.py
