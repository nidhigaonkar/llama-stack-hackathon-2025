from flask import Flask, render_template, jsonify, send_from_directory, request
import os

app = Flask(__name__)
BASE_DIR = os.path.abspath("input_folder")  # Update to your desired starting folder

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/list-files")
def list_files():
    subpath = request.args.get('path', '')
    abs_path = os.path.abspath(os.path.join(BASE_DIR, subpath))

    # Security: Prevent going outside BASE_DIR
    if not abs_path.startswith(BASE_DIR):
        return jsonify({"error": "Invalid path"}), 400

    entries = []
    for entry in os.listdir(abs_path):
        full_path = os.path.join(abs_path, entry)
        entries.append({
            "name": entry,
            "is_dir": os.path.isdir(full_path),
            "path": os.path.relpath(full_path, BASE_DIR)
        })

    return jsonify(entries)

@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)