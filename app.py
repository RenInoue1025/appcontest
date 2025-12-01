import os
import numpy as np
import itertools
from flask import Flask, render_template, request, url_for
from PIL import Image
from ultralytics import YOLO  # YOLOv8

app = Flask(__name__)
app.secret_key = "dummy_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================
# モデル（あなたの学習済みモデルパスに合わせて）
# =====================================
messy_model = YOLO("best.pt")
desk_model = YOLO("best.pt")

# --------------------------------------------
# パラメータ
# --------------------------------------------
CONF_THRESHOLD = 0.30
MIN_BOX_AREA = 0.001
PAPER_MIN_AREA = 0.002
OVERLAP_SCALE = 0.6
SCORE_TO_PERCENT = 3.0
DEBUG = False

ALLOWED_CLASSES = {
    "keyboard", "mouse", "laptop", "remote", "cell phone",
    "book", "cup", "bottle", "chair", "tv", "monitor",
    "backpack", "handbag", "vase", "scissors", "paper"
}

NORMAL_ITEMS_LIMITS = {
    "keyboard": 1,
    "mouse": 1,
    "laptop": 1,
    "monitor": 2,
    "chair": 1,
    "cup": 1,
    "bottle": 1,
}

GARBAGE_WEIGHTS = {
    "book": 2,
    "paper": 1,
    "cell phone": 2,
    "remote": 2,
    "backpack": 2,
    "handbag": 2,
    "banana": 3,
    "apple": 2,
    "teddy bear": 2,
}
DEFAULT_WEIGHT = 1


# ==========================================================
# IoU
# ==========================================================
def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_overlap_score(boxes):
    score = 0.0
    for b1, b2 in itertools.combinations(boxes, 2):
        iou = iou_xyxy(b1, b2)
        if iou > 0.7:
            score += 5.0 * OVERLAP_SCALE
        elif iou > 0.5:
            score += 3.0 * OVERLAP_SCALE
        elif iou > 0.3:
            score += 1.0 * OVERLAP_SCALE
    return score


# ==========================================================
# ★ 画像縮小処理（追加）
# ==========================================================
def resize_image_if_needed(input_path, max_width=1280):
    """
    画像の横幅が max_width を超える場合に縮小した画像を保存し、
    新しい一時ファイルのパスを返す。
    モデル推論にはこの縮小画像を使用する。
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    if w <= max_width:
        return input_path  # 縮小不要

    scale = max_width / w
    new_size = (max_width, int(h * scale))

    resized_img = img.resize(new_size, Image.LANCZOS)

    temp_path = input_path.replace(".jpg", "_resized.jpg").replace(".png", "_resized.png")
    resized_img.save(temp_path)

    return temp_path


# ==========================================================
# ルーティング
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return render_template("index.html", result="画像を受け取れませんでした。")

        # オリジナル保存
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ★ 画像縮小 → 推論用ファイルパス取得
        resized_path = resize_image_if_needed(filepath)

        # 画像サイズ取得
        pil_img = Image.open(resized_path).convert("RGB")
        img_w, img_h = pil_img.size
        img_area = float(img_w * img_h)

        # ---- (1) 机判定 ----
        desk_results = desk_model(resized_path, verbose=False)[0]
        desk_detected = False
        for box in desk_results.boxes:
            cls_id = int(box.cls)
            label = desk_model.names[cls_id]
            if label.lower() in ["desk", "table"]:
                desk_detected = True
                break

        if not desk_detected:
            return render_template(
                "index.html",
                result="机が写っている画像ではありません。もう一度撮影してください。",
                uploaded_image=url_for("static", filename=f"uploads/{filename}")
            )

        # ---- (2) 散らかり判定 ----
        detection = messy_model(resized_path, verbose=False)[0]

        object_count = {}
        boxes_abs = []
        paper_centers = []
        paper_count = 0

        for box in detection.boxes:
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = 0.0

            if conf < CONF_THRESHOLD:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = (w * h) / img_area

            if area < MIN_BOX_AREA:
                continue

            obj_name = messy_model.names[int(box.cls)]

            if obj_name == "paper" and area < PAPER_MIN_AREA:
                continue

            object_count[obj_name] = object_count.get(obj_name, 0) + 1
            boxes_abs.append([x1, y1, x2, y2])

            if obj_name == "paper":
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                paper_centers.append((cx, cy))
                paper_count += 1

        # ---- (3) スコア計算 ----
        messiness_score = 0.0
        display_objects = {}

        for obj, count in object_count.items():
            if obj in NORMAL_ITEMS_LIMITS:
                limit = NORMAL_ITEMS_LIMITS[obj]
                if count > limit:
                    excess = count - limit
                    messiness_score += excess
                    display_objects[obj] = excess
                continue

            if obj in GARBAGE_WEIGHTS:
                messiness_score += GARBAGE_WEIGHTS[obj] * count
                display_objects[obj] = count
                continue

            if obj in ALLOWED_CLASSES:
                messiness_score += DEFAULT_WEIGHT * count
                display_objects[obj] = count

        messiness_score += calculate_overlap_score(boxes_abs)

        # 紙散らばり
        spread_score = 0
        if len(paper_centers) >= 3:
            xs = [c[0] for c in paper_centers]
            ys = [c[1] for c in paper_centers]
            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)
            spread_area = spread_x * spread_y
            if spread_area > 0.25:
                spread_score = 8
            elif spread_area > 0.12:
                spread_score = 4
        messiness_score += spread_score

        # 紙の枚数
        if paper_count > 0:
            if paper_count <= 3:
                messiness_score += paper_count * 0.8
            elif paper_count == 4:
                messiness_score += paper_count * 1.2
            elif paper_count == 5:
                messiness_score += paper_count * 1.5
            elif paper_count <= 8:
                messiness_score += paper_count * 1.8
            else:
                messiness_score += paper_count * 2.5

        messiness_percent = min(100.0, messiness_score * SCORE_TO_PERCENT)

        # メッセージ生成
        message = f"あなたの机は <b>{messiness_percent:.1f}%</b> 散らかっています。<br>"
        if messiness_percent > 70:
            message += "かなり散らかっています。片付けましょう🧹<br>"
        elif messiness_percent > 40:
            message += "少し散らかっています。整理しましょう🪣<br>"
        else:
            message += "とても綺麗です✨<br>"

        if display_objects:
            detected_list = ", ".join([f"{o} × {c}" for o, c in display_objects.items()])
            message += f"散らかり要因：{detected_list}<br>"
        else:
            message += "散らかり要因となる物体は検出されませんでした。<br>"

        return render_template(
            "index.html",
            result=message,
            uploaded_image=url_for("static", filename=f"uploads/{filename}")
        )

    except Exception as e:
        print(f"❌ エラー: {e}")
        return render_template("index.html", result=f"エラーが発生しました: {e}")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/calendar")
def calendar():
    return render_template("calendar.html")


@app.route("/rewards")
def rewards():
    return render_template("rewards.html")


@app.route("/settings")
def settings():
    return render_template("settings.html")


if __name__ == "__main__":
    app.run(debug=True)
