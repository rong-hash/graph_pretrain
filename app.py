from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, ALL
import os
import threading

app = Flask(__name__)
app.config['UPLOADS_DEFAULT_DEST'] = 'uploads'
uploaded_files = UploadSet('files', ALL)
configure_uploads(app, uploaded_files)

def process_file(file_path, method):
    # Add your algorithm implementations here
    if method == "Node":
        pass  # Replace with your algorithm for Method 1
        return
    elif method == "Graph":
        pass  # Replace with your algorithm for Method 2
        return 
    else:
        raise ValueError("Invalid method")

    # Return the result or error message
    return "Result or Error Message"

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'file' in request.files:
        method = request.form['method']
        filename = uploaded_files.save(request.files['file'])
        file_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], 'files', filename)
        
        result = process_file(file_path, method)
        
        # Remove the file after processing
        os.remove(file_path)
        
        return jsonify({'result': result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
