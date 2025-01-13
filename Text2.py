import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# Инициализация модели
tokenizer, model = load_model()

# Интерфейс приложения
st.title("Генерация текста по смыслу слова")
st.write("Введите слово, чтобы получить небольшой текст, связанный с его смыслом.")

# Поле для ввода слова
input_word = st.text_input("Введите слово на русском языке:")

# Кнопка для генерации текста
if st.button("Сгенерировать текст"):
    word = input_word.strip().lower()
    if word:
        try:
            # Генерация текста
            input_ids = tokenizer.encode(word, return_tensors="pt")
            output = model.generate(
                input_ids,
                max_length=50,  # Длина генерируемого текста
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Вывод результата
            st.subheader("Сгенерированный текст:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"Ошибка при генерации текста: {e}")
    else:
        st.warning("Пожалуйста, введите слово для генерации текста.")
