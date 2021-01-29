from flask import Flask
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import re
from flask import request
from io import StringIO
from io import BytesIO

app = Flask(__name__)
app.secret_key = "markus"
CORS(app)


@app.route("/", methods=["POST"])
def get_image():
    result = request.data

    print(result)
    return ""


if __name__ == "__main__":
    app.run(port=5000, debug=True)
