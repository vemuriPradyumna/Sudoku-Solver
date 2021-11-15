import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from alda_image_pp import preprocess, puzzle, solve, Suduko 
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("C:\\Users\\sudha\\OneDrive - North Carolina State University\\Documents\\ALDA\\sudoku-solver\\TMINSTmodel.h5")

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'images'


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        uploaded_file.save(path)
        grid = preprocess(path)
        if (Suduko(grid, 0, 0)):
            puzzle(grid)
        else:
            print("Solution does not exist:(")
    
    return redirect(url_for('index'))