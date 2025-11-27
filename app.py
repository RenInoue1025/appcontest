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
messy_model = YOLO("runs/detect/desk_paper_book/weights/best.pt")
desk_model = YOLO("combined_dataset/runs/detect/train/weights/best.pt")

# --------------------------------------------
# パラメータ（調整しやすいようここに集約）
# --------------------------------------------
CONF_THRESHOLD = 0.30       # 検出の最低信頼度（高めにすると誤検出減る）
MIN_BOX_AREA = 0.001        # 無視する極小BBox（画像比率で、0.001=0.1%）
PAPER_MIN_AREA = 0.002      # paper と認める最小面積（誤検出抑制）
OVERLAP_SCALE = 0.6         # IoU による加点のスケーリング（0.0〜1.0）
SCORE_TO_PERCENT = 3.0      # 内部スコア -> % にする係数（小さくすると厳しくない）
DEBUG = False               # True にすると内部スコアを結果表示する

# --------------------------------------------
# 物体判定ルール
# --------------------------------------------
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
    "paper": 1,        # ← 下げた（paperの誤検出による影響を抑制）
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
# IoU（重なり度）を計算（手書き実装）
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return render_template("index.html", result="画像を受け取れませんでした。")

        # 保存
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 画像サイズ取得（正規化面積計算のため）
        pil_img = Image.open(filepath).convert("RGB")
        img_w, img_h = pil_img.size
        img_area = float(img_w * img_h)

        # ---- (1) 机判定（専用モデル） ----
        desk_results = desk_model(filepath, verbose=False)[0]
        desk_detected = False
        for box in desk_results.boxes:
            cls_id = int(box.cls)
            label = desk_model.names[cls_id]
            if label.lower() in ["desk", "table"]:
                desk_detected = True
                break

        if not desk_detected:
            # ユーザー修正（あとでボタンで「これは机です」を追加する設計にできます）
            return render_template(
                "index.html",
                result="机が写っている画像ではありません。もう一度撮影してください。",
                uploaded_image=url_for("static", filename=f"uploads/{filename}")
            )

        # ---- (2) 散らかり判定（messy_model） ----
        detection = messy_model(filepath, verbose=False)[0]

        # 集計用
        object_count = {}
        boxes_norm = []   # 正規化された xyxy (0..1)
        boxes_abs = []    # 絶対座標（ピクセル）
        paper_centers = []
        paper_count = 0

        for box in detection.boxes:
            # confidence を扱えるように安全に取得
            try:
                conf = float(box.conf[0])
            except Exception:
                # 互換性のため別の取り方
                conf = float(getattr(box, "conf", 0.0) if hasattr(box, "conf") else 0.0)

            # スコアが低ければ無視
            if conf < CONF_THRESHOLD:
                continue

            # bbox 座標
            xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2] (pixels)
            x1, y1, x2, y2 = map(float, xyxy)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = (w * h) / max(1.0, img_area)  # 正規化面積 (0..1)

            # 小さすぎる箱はノイズとみなす
            if area < MIN_BOX_AREA:
                continue

            obj_name = messy_model.names[int(box.cls)]

            # paper は面積が小さければ無視（誤検出抑制）
            if obj_name == "paper" and area < PAPER_MIN_AREA:
                continue

            # 集計
            object_count[obj_name] = object_count.get(obj_name, 0) + 1
            boxes_norm.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
            boxes_abs.append([x1, y1, x2, y2])

            if obj_name == "paper":
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                paper_centers.append((cx, cy))
                paper_count += 1

        # ---- (3) スコア計算 ----
        messiness_score = 0.0
        display_objects = {}

        # 基本の件数重み
        for obj, count in object_count.items():
            if obj in NORMAL_ITEMS_LIMITS:
                limit = NORMAL_ITEMS_LIMITS[obj]
                if count > limit:
                    excess = count - limit
                    messiness_score += excess * 1.0
                    display_objects[obj] = excess
                # 常設アイテムはここまで
                continue

            if obj in GARBAGE_WEIGHTS:
                messiness_score += GARBAGE_WEIGHTS[obj] * count
                display_objects[obj] = count
                continue

            if obj in ALLOWED_CLASSES:
                messiness_score += DEFAULT_WEIGHT * count
                display_objects[obj] = count

        # 重なり（IoU）スコア（boxes_abs を使用）
        overlap_score = calculate_overlap_score(boxes_abs)
        messiness_score += overlap_score

        # 紙の散らばり（paper_centers の範囲）→加点
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

        # 紙の枚数ペナルティ（指数的）
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

        # 正規化（内部スコア->%）
        messiness_percent = min(100.0, messiness_score * SCORE_TO_PERCENT)

        # メッセージ組み立て
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

        if DEBUG:
            message += f"(デバッグ) conf_th={CONF_THRESHOLD}, min_area={MIN_BOX_AREA}, paper_count={paper_count}, overlap={overlap_score:.2f}, spread={spread_score}, raw_score={messiness_score:.2f})"

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
