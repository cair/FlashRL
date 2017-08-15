from flask import Flask, request, send_from_directory
import os
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')


@app.route('/env/<path:path>')
def send(path):
    return send_from_directory(os.path.join(os.getcwd(), 'environments'), path, mimetype="application/x-shockwave-flash")


