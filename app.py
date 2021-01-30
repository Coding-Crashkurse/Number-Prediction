from flask import Flask
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import re
from flask import request
from io import StringIO
from io import BytesIO
import ast
import torch
from torch import nn
import torch.nn.functional as F
from flask import jsonify


model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1),
)
model.load_state_dict(torch.load("./model.pth"))
model.eval()


def convert_scale(arr):
    result = -1 + 2 * ((arr) - np.min(arr)) / (np.max(arr) - np.min(arr))
    return result


app = Flask(__name__)
app.secret_key = "markus"
CORS(app)


@app.route("/", methods=["GET", "POST"])
def get_image():
    result = ast.literal_eval(request.data.decode("utf-8"))
    result_arr = np.array(result.get("alpha_arr"))
    result_arr = convert_scale(result_arr)
    T = torch.from_numpy(result_arr.astype(np.float32))
    T = T.view((1, 420, 420))
    T = T.unsqueeze(0)
    T = torch.nn.functional.interpolate(T, size=(28, 28), mode="bilinear")
    T = T.squeeze(0)
    img = T.view(1, 784)
    with torch.no_grad():
        logits = model.forward(img)

    ps = F.softmax(logits, dim=1)
    ps_num = ps.numpy().flatten()

    return jsonify(prob=float(ps_num.max()), prediction=int(ps_num.argmax()))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
