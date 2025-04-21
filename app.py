from flask import Flask, render_template, jsonify, send_from_directory
import os

app = Flask(__name__)
FILE_DIRECTORY = os.path.abspath("input_folder")  # Change "your_folder" to where your files are

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/list-files")
def list_files():
    files = os.listdir(FILE_DIRECTORY)
    return jsonify(files)

@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(FILE_DIRECTORY, filename)

if __name__ == "__main__":
    app.run(debug=True)