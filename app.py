from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from predict import predict_mood

app = Flask(__name__)

# ✅ Upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "❌ No image part in the request"

        file = request.files['image']

        if file.filename == '':
            return "❌ No file selected"

        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"✅ Saving image to: {save_path}")
            file.save(save_path)

            mood = predict_mood(save_path)
            return render_template('result.html', prediction=mood, image=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
