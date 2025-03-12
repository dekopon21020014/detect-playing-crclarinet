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
face_measure_buffer = []  # 各フレームでの「顔の位置」と肩との相対位置を格納

# 閾値（単位：ピクセル、環境に合わせて調整）
DOWN_THRESHOLD = 50   # 顔が baseline より下に大きくずれたと判断
UP_THRESHOLD   = -10  # 顔が baseline より上にずれたと判断

# 状態機： "IDLE"（待機状態）、"DOWN"（下向き検出済）、"UP"（上向き検出済）
state = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ミラー表示
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe 用に BGR→RGB 変換
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.face_landmarks and results.pose_landmarks:
        # 顔ランドマークから顎と鼻の位置を取得（正規化座標→ピクセル変換）
        face_landmarks = results.face_landmarks.landmark
        jaw_x = face_landmarks[152].x * w
        jaw_y = face_landmarks[152].y * h
        nose_x = face_landmarks[1].x * w
        nose_y = face_landmarks[1].y * h
        # 顎と鼻の平均位置（顔の下部の代表値とする）
        face_y = (jaw_y + nose_y) / 2

        # 姿勢ランドマークから左右肩を取得（Pose モジュール: 11: 左肩、12: 右肩）
        pose_landmarks = results.pose_landmarks.landmark
        left_shoulder_y = pose_landmarks[11].y * h
        right_shoulder_y = pose_landmarks[12].y * h
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        # 顔の位置と肩との相対位置（値が大きいほど顔が下にある）
        relative_face = face_y - shoulder_y

        # スライディングウィンドウに追加して baseline を計算
        face_measure_buffer.append(relative_face)
        if len(face_measure_buffer) > window_size:
            face_measure_buffer.pop(0)
        baseline = np.mean(face_measure_buffer)

        # baseline との差（deviation）
        deviation = relative_face - baseline

        # 顔のランドマークおよび肩の位置を描画
        cv2.circle(image_rgb, (int(jaw_x), int(jaw_y)), 3, (0, 0, 255), -1)    # 顎: 赤
        cv2.circle(image_rgb, (int(nose_x), int(nose_y)), 3, (0, 255, 0), -1)    # 鼻: 緑
        left_shoulder_x = pose_landmarks[11].x * w
        right_shoulder_x = pose_landmarks[12].x * w
        cv2.circle(image_rgb, (int(left_shoulder_x), int(left_shoulder_y)), 3, (255, 0, 0), -1)
        cv2.circle(image_rgb, (int(right_shoulder_x), int(right_shoulder_y)), 3, (255, 0, 0), -1)

        cv2.putText(image_rgb, f"RelFace: {relative_face:.2f} (dev: {deviation:.2f})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 状態機による検知（下→上→下の動作で吹き始めと判断）
        if state == "IDLE" and deviation > 70:
            state = "DOWN"
            print("DOWN state detected")
        elif state == "DOWN" and deviation < UP_THRESHOLD:
            state = "UP"
            print("UP state detected")
        elif state == "UP" and deviation > DOWN_THRESHOLD:
            detection_timestamp = time.time()
            print("Blow Start Detected Timestamp:", detection_timestamp)
            threading.Thread(target=display_detection_message, args=(5,), daemon=True).start()
            state = "IDLE"

        # baseline（参考用）の描画：肩を基準とした水平線
        baseline_abs = baseline + shoulder_y
        cv2.line(image_rgb, (0, int(baseline_abs)), (w, int(baseline_abs)), (0, 255, 0), 1)

    # 常時 state を画面に表示
    cv2.putText(image_rgb, f"State: {state}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 検知中なら検知メッセージを表示
    if detection_display_active and detection_timestamp is not None:
        cv2.putText(image_rgb, f"Blow Start Detected: {detection_timestamp:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Clarinet Blow Detection - Whole Body", image_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()
