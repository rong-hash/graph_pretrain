from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import threading

app = Flask(__name__)
app.config['UPLOADS_DEFAULT_DEST'] = 'uploads/'

def process_file(predata_path, downdata_path, method):
    # Implement your algorithm here
    result = "Processed files: {} and {} using method {}".format(predata_path, downdata_path, method)
    return result

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'predata' in request.files and 'downdata' in request.files:
        method = request.form['method']
        predata = request.files['predata']
        downdata = request.files['downdata']
        predata_filename = secure_filename(predata.filename)
        downdata_filename = secure_filename(downdata.filename)
        predata_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], predata_filename)
        downdata_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], downdata_filename)
        predata.save(predata_path)
        downdata.save(downdata_path)

        result = process_file(predata_path, downdata_path, method)

        # Remove the files after processing
        os.remove(predata_path)
        os.remove(downdata_path)

        return jsonify({'result': result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
