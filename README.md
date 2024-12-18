# Hackathon Site

Этот проект представляет собой веб-приложение на Flask с интегрированным чат-ботом для обработки вопросов и ответов из предоставленного датасета. 

## Структура проекта

1. Скачайте файлы `main.py`, `app.py` и папку `templates` с HTML-файлом для фронтенда.
2. Все файлы должны находиться в одной директории:
├── app.py ├── main.py ├── templates/ └── index.html

## Запуск проекта

1. **Запустите файл `app.py`.**  
При запуске создается локальный сервер с веб-интерфейсом чат-бота.

2. **Загрузите файл для обработки:**  
- Нажмите кнопку **"Выберите файл"** и выберите файл `test.docx`.
- Нажмите кнопку **"Загрузить файл"**. Дождитесь уведомления:  
  `Файл test.docx успешно загружен`.

3. **Обучите модель:**  
- Нажмите кнопку **"Обучить модель"**.  
  В редакторе кода начнется процесс обучения модели (используется 300 эпох).  
- После завершения обучения появится уведомление:  
  `Модель успешно обучена`.

4. **Задайте вопросы:**  
Введите любой вопрос из файла `test.docx`, и модель предоставит правильный ответ.

---

## Особенности модели

- Наша выборка состоит из 500 записей. Для увеличения объема датасета мы искусственно добавили по 5 вопросов для каждого ответа. Это сделано с помощью цикла в функции **prepare_data**.

 
```python
for line in lines:
    line = line.strip()
    if line.endswith("?"):
        questions.append(line)
    elif line.endswith("."):
        answer = clean_string(line)
        for _ in range(5):  # Дублируем каждый ответ 5 раз
            answers.append(answer)
```

## Как изменить код под свою модель
- Если вы хотите протестировать собственный датасет, измените код обработки данных в файле main.py.
- Если ваш датасет имеет другой формат (например, JSON-объекты), используйте следующий пример:

```python
# Пример датасета
data = [
    {"question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes"},
    {"question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", "answer": "Chief of Protocol"},
    {"question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?", "answer": "Animorphs"},
    {"question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?", "answer": "no"},
    {"question": "The director of the romantic comedy 'Big Stone Gap' is based in what New York city?", "answer": "Greenwich Village, New York City"}
]

# Инициализация списков
questions = []
answers = []

# Извлечение данных
for entry in data:
    questions.append(entry["question"])
    answers.append(entry["answer"])
```

После выполнения этого кода:
- questions будет содержать все вопросы.
- answers будет содержать все ответы.

