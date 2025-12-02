import os
import itertools
from flask import Flask, render_template, request, url_for
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "dummy_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================
# 学習済みモデル
# =====================================
messy_model = YOLO("best.pt")

# --------------------------------------------
# パラメータ
# --------------------------------------------
CONF_THRESHOLD = 0.30
MIN_BOX_AREA = 0.001
PAPER_MIN_AREA = 0.002
OVERLAP_SCALE = 0.6
SCORE_TO_PERCENT = 3.0

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
# 散らかり度の段階
# ==========================================================
MESSINESS_LABELS = [
    (0, 20, "A：とても綺麗ですね！ この調子をキープしましょう。"),
    (20, 40, "B：少し物が増えてきました。1〜2 個片付けるとスッキリします。"),
    (40, 60, "C：散らかり気味です。よく使わない物から片付けていきましょう。"),
    (60, 80, "D：かなり散らかっています！「分類 → 収納 → 捨てる」を意識しましょう。"),
    (80, 100, "E：危険レベル！一度リセットするつもりで大掃除をおすすめします。")
]

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
# 画像縮小処理
# ==========================================================
def resize_image_if_needed(input_path, max_width=1280):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    if w <= max_width:
        return input_path
    scale = max_width / w
    resized_img = img.resize((max_width, int(h * scale)), Image.LANCZOS)
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
    file = request.files.get("file")
    if not file:
        return render_template("index.html", result="画像を受け取れませんでした。")

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    resized_path = resize_image_if_needed(filepath)

    # 画像サイズ
    pil_img = Image.open(resized_path).convert("RGB")
    img_w, img_h = pil_img.size
    img_area = float(img_w * img_h)

    # ---- 散らかり判定 ----
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

    # ---- スコア計算 ----
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

    # 紙の散らばり
    if len(paper_centers) >= 3:
        xs = [c[0] for c in paper_centers]
        ys = [c[1] for c in paper_centers]
        spread_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if spread_area > 0.25:
            messiness_score += 8
        elif spread_area > 0.12:
            messiness_score += 4

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

    # 段階判定
    level_message = ""
    for low, high, text in MESSINESS_LABELS:
        if low <= messiness_percent < high:
            level_message = text
            break

    if display_objects:
        detected_list = ", ".join([f"{o} × {c}" for o, c in display_objects.items()])
        level_message += f"<br>散らかり要因：{detected_list}<br>"

    return render_template(
        "index.html",
        result=f"あなたの机は <b>{messiness_percent:.1f}%</b> 散らかっています。<br>{level_message}",
        uploaded_image=url_for("static", filename=f"uploads/{filename}")
    )


# --- その他ページ ---
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
    app.run(host="0.0.0.0", port=5000, debug=True)
