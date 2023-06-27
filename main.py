from flask import Flask, request, redirect, url_for, render_template, Markup, make_response
from werkzeug.utils import secure_filename
from io import BytesIO
import io
import base64

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
#from PIL import Image

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
import torchvision.transforms as T

torch.set_grad_enabled(False)  # 訓練は行わないので勾配の計算は不要

import os
import shutil

# データセットCOCOの物体名
names = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
    "truck", "boat", "traffic light", "fire hydrant", "N/A","stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack","umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 読込ファイル種別チェック
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# バウンディングボックスの座標変換
def cxcywh_to_4corners(x):
    x_c, y_c, w, h = x.unbind(1)
    box = [(x_c - 0.5 * w),
           (y_c - 0.5 * h),
           (x_c + 0.5 * w),
           (y_c + 0.5 * h)]
    return torch.stack(box, dim=1)

# バウンディングボックスのスケール変換
def fit_boxes(y_box, size):
    w, h = size
    box = cxcywh_to_4corners(y_box)
    box = box * torch.tensor([w, h, w, h], dtype=torch.float)
    return box

# 結果の表示
def show_results(img, ps, boxes):

    boxes = boxes.tolist()
    
    fig = plt.figure(figsize=(16,10))
    plt.imshow(img)

    ax = plt.gca()
    for p, (x_min, y_min, x_max, y_max) in zip(ps, boxes):
        ax.add_patch(plt.Rectangle((x_min, y_min),
                                   x_max - x_min,
                                   y_max - y_min,
                                   fill=False,
                                   color="red",
                                   linewidth=3))
        
        result_id = p.argmax()
        label = f"{names[result_id]}: {p[result_id]:0.3f}"
        ax.text(
            x_min,y_min,
            label, fontsize=12,
            bbox=dict(facecolor="orange", alpha=0.4)
            )
        
    plt.axis("off")
    base64_img = fig_to_base64_img(fig)
    return base64_img

# -----figをbase64形式に変換を関数化------
def fig_to_base64_img(fig):
    io = BytesIO()
    fig.savefig(io, format="png")
    io.seek(0)
    base64_img = base64.b64encode(io.read()).decode()

    return base64_img

# モデルの読込
model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True)
model.eval()  # 評価モード

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        img_origin = Image.open(filepath)

        # 画像の変換
        transform = T.Compose([
            T.Resize(800),  # 短い辺を800に変換
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
        ])
        x = transform(img_origin).unsqueeze(0)  # unsqueezeでバッチ対応

        # 予測
        y = model(x)

        # 予測結果の選別
        ps = y["pred_logits"].softmax(-1)[0, :, :-1]
        extracted = ps.max(-1).values > 0.90 # 0.95より確率が大きいものを選別

        # バウンディングボックスの座標計算
        boxes = fit_boxes(y["pred_boxes"][0, extracted], img_origin.size)

        # 予測結果の表示
        img = show_results(img_origin, ps[extracted], boxes)

        return render_template("result.html", img = img)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
