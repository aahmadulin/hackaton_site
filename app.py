from flask import Flask, render_template, request, jsonify
import os
import docx2txt
import PyPDF2
import json
from main import train_model_on_data, predict_answer  # Импорт из main.py

app = Flask(__name__)

# Папка для загрузки файлов
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'json'}

# Создаем папку для загрузок
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Проверка расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Переменная для хранения загруженного текста
global user_data
user_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global user_data

    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла для загрузки'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Нет выбранного файла'})

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Обрабатываем файл, извлекая текст
        extracted_text = ''
        if 'doc' in filename:  # DOCX
            extracted_text = docx2txt.process(filepath)
        elif 'pdf' in filename:  # PDF
            with open(filepath, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                extracted_text = ''.join([page.extract_text() for page in reader.pages])
        elif 'json' in filename:  # JSON
            with open(filepath, encoding='utf-8') as json_file:
                data = json.load(json_file)
                extracted_text = json.dumps(data, ensure_ascii=False, indent=4)

        # Сохраняем данные для обучения
        user_data = extracted_text
        return jsonify({'success': f'Файл {filename} успешно загружен', 'content': extracted_text})

    return jsonify({'error': 'Недопустимый формат файла'})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        print("Начинаю обучение модели...")
        success = train_model_on_data(user_data)
        if success:
            print("Модель обучена успешно.")
            return jsonify({"success": "Модель успешно обучена."})
        else:
            print("Ошибка при обучении модели.")
            return jsonify({"error": "Ошибка обучения модели."})
    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({"error": str(e)})


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message')

    # Получаем ответ от модели
    response = predict_answer(user_message)
    return jsonify({'response': response})

    # data = request.json
    # question = data.get("message", "")
    # response = predict_answer(question)
    # return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
