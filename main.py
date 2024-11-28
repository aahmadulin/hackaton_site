import tensorflow as tf
import keras
import numpy as np

# Глобальные переменные
model = None
vectorize_layer = None
answer_map = {}  # Словарь {номер класса: текст ответа}

# Параметры
max_tokens = 10000
max_len = 100

# Функция для очистки текста
def clean_string(input_string):
    cleaned_string = input_string.split(". ", 1)[-1]  # Берем часть после первой точки
    cleaned_string = cleaned_string.strip()  # Убираем лишние пробелы
    return cleaned_string

# Функция для подготовки данных
def prepare_data(raw_text):
    lines = raw_text.split("\n")
    questions, answers = [], []

    for line in lines:
        line = line.strip()
        if line.endswith("?"):
            questions.append(line)
        elif line.endswith("."):
            answer = clean_string(line)
            # Дублируем каждый ответ 5 раз
            for _ in range(5):
                answers.append(answer)

    # Проверяем, чтобы длины были корректными
    if len(answers) != len(questions):
        raise ValueError(f"Количество вопросов ({len(questions)}) не совпадает с количеством ответов ({len(answers)})!")
    
    return questions, answers

# Функция для обучения модели
def train_model_on_data(raw_text):
    global model, vectorize_layer, answer_map

    try:
        # Подготовка данных
        questions, answers = prepare_data(raw_text)

        # Создаем маппинг {ответ: индекс}
        unique_answers = list(set(answers))  # Уникальные ответы
        answer_map = {answer: idx for idx, answer in enumerate(unique_answers)}
        labels = np.array([answer_map[answer] for answer in answers])

        # Обратный маппинг для предсказаний
        answer_map = {idx: answer for answer, idx in answer_map.items()}

        # Создаем TextVectorization
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=max_len
        )
        vectorize_layer.adapt(questions + answers)

        # Преобразуем текст в векторы
        vectorized_questions = vectorize_layer(questions)

        # Создаем модель
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,), dtype='int32'),
            tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=64),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(unique_answers), activation='softmax')
        ])

        # Компиляция и обучение модели
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(vectorized_questions, labels, epochs=300, batch_size=32, validation_split=0.2)

        # Сохраняем модель
        keras.saving.save_model(model, 'my_model.keras')
        return True
    except Exception as e:
        print(f"Ошибка обучения модели: {e}")
        return False

# Функция для предсказания ответа
def predict_answer(question):
    global model, vectorize_layer, answer_map

    if not model or not vectorize_layer:
        return "Модель еще не обучена. Сначала загрузите данные и обучите модель."

    vectorized_question = vectorize_layer([question])
    predictions = model.predict(vectorized_question)
    predicted_class = np.argmax(predictions)
    return answer_map.get(predicted_class, "Ответ не найден")
