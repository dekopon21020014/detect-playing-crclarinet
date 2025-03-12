import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# 非同期で検知メッセージを表示するためのグローバル変数
detection_display_active = False
detection_timestamp = None

def display_detection_message(duration=5):
    """
    検知が行われたことを示すフラグをオンにし、指定時間後にオフにする関数（非同期呼び出し用）。
    """
    global detection_display_active
    detection_display_active = True
    time.sleep(duration)
    detection_display_active = False

# MediaPipe Holistic の初期化（全身＋顔のランドマーク）
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# 複数フレームの測定値を保存するためのスライディングウィンドウ
window_size = 10
face_measure_buffer = []  # 各フレームでの「顔の位置」と「肩との相対位置」の値を格納

# ここでは、顔の「下部（顎と鼻の平均）」と左右肩の平均との縦方向の差を計算
# ※ y座標は画像の上から下へ向かって増加するため、値が大きいほど顔が下に位置する
# 閾値（単位：ピクセル、環境により調整が必要）
DOWN_THRESHOLD = 15   # 顔が baseline より下に大きくずれたと判断
UP_THRESHOLD   = -10  # 顔が baseline より上にずれたと判断

# 状態機： "IDLE"（待機状態）、"DOWN"（下向き状態検出済）、"UP"（上向き状態検出済）
state = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ミラー表示
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe 用に BGR -> RGB 変換
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.face_landmarks and results.pose_landmarks:
        # 顔ランドマーク（MediaPipe Face Mesh 468点）から顎と鼻を取得
        face_landmarks = results.face_landmarks.landmark

        # 例として、顎：インデックス 152、鼻：インデックス 1 を使用（座標は正規化されている）
        jaw_x = face_landmarks[152].x * w
        jaw_y = face_landmarks[152].y * h
        nose_x = face_landmarks[1].x * w
        nose_y = face_landmarks[1].y * h

        # 顎と鼻の平均位置（顔の下部付近とみなす）
        face_y = (jaw_y + nose_y) / 2

        # 姿勢ランドマークから左右肩を取得（Pose モジュールのインデックス 11：左肩、12：右肩）
        pose_landmarks = results.pose_landmarks.landmark
        left_shoulder_y = pose_landmarks[11].y * h
        right_shoulder_y = pose_landmarks[12].y * h
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        # 顔の位置と肩の位置との差（大きいほど顔が下に位置している）
        relative_face = face_y - shoulder_y

        # スライディングウィンドウに追加し、baseline を計算
        face_measure_buffer.append(relative_face)
        if len(face_measure_buffer) > window_size:
            face_measure_buffer.pop(0)
        baseline = np.mean(face_measure_buffer)

        # 現在の測定値と baseline の差（deviation）
        deviation = relative_face - baseline

        # デバッグ用に各ポイントを描画
        cv2.circle(image_rgb, (int(jaw_x), int(jaw_y)), 3, (0, 0, 255), -1)    # 顎：赤
        cv2.circle(image_rgb, (int(nose_x), int(nose_y)), 3, (0, 255, 0), -1)    # 鼻：緑
        # 肩
        left_shoulder_x = pose_landmarks[11].x * w
        right_shoulder_x = pose_landmarks[12].x * w
        cv2.circle(image_rgb, (int(left_shoulder_x), int(left_shoulder_y)), 3, (255, 0, 0), -1)
        cv2.circle(image_rgb, (int(right_shoulder_x), int(right_shoulder_y)), 3, (255, 0, 0), -1)

        cv2.putText(image_rgb, f"RelFace: {relative_face:.2f} (dev: {deviation:.2f})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 状態機による検知
        if state == "IDLE" and deviation > DOWN_THRESHOLD:
            state = "DOWN"
            print("DOWN 状態検出")
        elif state == "DOWN" and deviation < UP_THRESHOLD:
            state = "UP"
            print("UP 状態検出")
        elif state == "UP" and deviation > DOWN_THRESHOLD:
            # 下→上→下の動作が完了した瞬間と判断
            detection_timestamp = time.time()
            print("吹き始め検出タイムスタンプ:", detection_timestamp)
            threading.Thread(target=display_detection_message, args=(5,), daemon=True).start()
            state = "IDLE"

        # baseline（参考用）の描画（画面上に水平線を描く）
        baseline_abs = baseline + shoulder_y  # baseline の絶対位置
        cv2.line(image_rgb, (0, int(baseline_abs)), (w, int(baseline_abs)), (0, 255, 0), 1)

    # 非同期検知表示中ならメッセージを描画
    if detection_display_active and detection_timestamp is not None:
        cv2.putText(image_rgb, f"Blow Start Detected: {detection_timestamp:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Clarinet Blow Detection - Whole Body", image_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()
