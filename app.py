from flask import Flask, render_template, jsonify, send_from_directory, request, abort
import os
import subprocess

app = Flask(__name__)
BASE_DIR = os.path.expanduser("~")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/list-files')
def list_files():
    path = request.args.get('path', '')
    full_path = os.path.abspath(os.path.join(BASE_DIR, path))

    if not os.path.commonpath([BASE_DIR, full_path]).startswith(BASE_DIR):
        abort(403)

    entries = []
    for entry in os.scandir(full_path):
        if entry.name == '.DS_Store':
            continue

        entry_path = os.path.relpath(entry.path, BASE_DIR)
        entry_info = {
            'name': entry.name,
            'path': entry_path,
            'is_dir': entry.is_dir()
        }

        # Add preview for text/code files
        if entry.is_file() and entry.name.split('.')[-1].lower() in ['txt', 'md', 'py', 'js', 'html', 'css']:
            try:
                with open(entry.path, 'r', encoding='utf-8') as f:
                    preview = ''.join([next(f) for _ in range(5)])  # First 5 lines
                entry_info['preview'] = preview
            except Exception:
                entry_info['preview'] = '[Preview unavailable]'

        entries.append(entry_info)

    return jsonify(entries)

@app.route('/optimize-storage', methods=['POST'])
def optimize_storage():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    file_summary = data.get('file_summary', '')

    full_prompt = f"""
You are a file system optimizer. Given the following local file summary and the user's prompt, suggest the most efficient way to clean up their storage. Be concise and safe.

User Prompt: {user_prompt}

File Summary:
{file_summary}

Respond with a markdown-formatted list of actions the user should take.
"""

    try:
        result = subprocess.run(
            ['ollama', 'run', 'mistral'],
            input=full_prompt,
            text=True,
            capture_output=True,
            timeout=120
        )
        return jsonify({'response': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/files/<path:filename>")
def serve_file(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)