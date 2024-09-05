import math
import threading
import tkinter as tk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import spacy
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from paddleocr import PaddleOCR
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 環境変数の設定
nlp = spacy.load('ja_core_news_lg')  # SpaCyの日本語モデルを読み込み
ocr_to_accepted_words = {
    "秩父の天然水": {'秩父の天然水', '父の天然水', '稚父の天然水', '种父の天然水', '地父の天然水', '稚父の天然木',
                     '地父の天然木', '秩父の天然'}
}


def automatic_gaussian_blur(image):
    if len(image.shape) == 2:  # 画像がグレースケールの場合
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # グレースケールからBGRに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールに変換
    kernel_size = math.floor(gray.shape[1] / 20)  # カーネルサイズの計算
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # カーネルサイズを奇数に調整
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # ガウスぼかしの適用
    return blurred


def draw_text(image, text, position):
    img_pil = Image.fromarray(image)  # NumPy配列からPIL画像に変換
    draw = ImageDraw.Draw(img_pil)  # 描画オブジェクトの作成
    font_path = 'M_PLUS_1p/MPLUS1p-Regular.ttf'  # フォントファイルのパス
    font = ImageFont.truetype(font_path, 20)  # フォントの設定
    draw.text(position, text, font=font, fill=(0, 0, 255, 0))  # 画像にテキストを描画
    return np.array(img_pil)  # PIL画像をNumPy配列に変換して返す


def calculate_iou(box1, box2):
    xi1, yi1, xi2, yi2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[
        3])  # IOUの計算のための座標の計算
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)  # 交差部分の面積の計算
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  # ボックス1の面積の計算
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  # ボックス2の面積の計算
    union_area = box1_area + box2_area - inter_area  # 結合部分の面積の計算
    return inter_area / union_area  # IOUを返す


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window  # Tkinterウィンドウの設定
        self.window.title(window_title)  # ウィンドウのタイトル設定
        self.video_source = video_source  # ビデオソースの設定
        self.vid = None  # ビデオキャプチャオブジェクトの初期化
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang='japan')  # PaddleOCRの日本語モデルの設定
        self.result_label = tk.Label(window, text="OCR Result")  # OCR結果を表示するラベルの作成
        self.result_label.pack()  # ラベルをウィンドウに配置
        self.camera_open = False  # カメラのオープン状態の初期化
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # スレッドプールの設定

        # ボタンの設定と配置
        self.buttons = {
            "Open Camera": self.open_camera,
            "Snapshot": self.take_snapshot,
            "OCR": self.perform_ocr,
            "Exit": self.close
        }

        for button_text, button_command in self.buttons.items():
            tk.Button(window, text=button_text, width=20, command=button_command).pack()  # ボタンの作成と配置

        self.canvas = tk.Canvas(self.window, width=640, height=480)  # 画像を表示するためのキャンバスの作成
        self.canvas.pack()  # キャンバスをウィンドウに配置

        self.delay = 35  # 更新の遅延時間の設定
        self.update()  # 初回の更新呼び出し

    def open_camera(self):
        if not self.camera_open:  # カメラがまだオープンしていない場合
            self.vid = cv2.VideoCapture(self.video_source)  # ビデオキャプチャオブジェクトの作成
            if self.vid.isOpened():  # ビデオキャプチャが成功した場合
                self.camera_open = True  # カメラがオープンした状態に設定
                self.canvas.config(width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                   height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # キャンバスのサイズをビデオフレームのサイズに設定

    def take_snapshot(self):
        if self.camera_open:  # カメラがオープンしている場合
            ret, frame = self.vid.read()  # フレームの読み取り
            if ret:  # フレームが正常に読み取られた場合
                self.img_bgr = frame.copy()  # BGR画像のコピー
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))  # PIL画像からTkinterのPhotoImageに変換
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # キャンバスに画像を表示

                plt.figure("Snapshot")  # 新しい図を作成
                plt.imshow(self.img_rgb)  # RGB画像を表示
                plt.axis('off')  # 軸を非表示に設定
                plt.show()  # 画像を表示

    def perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):  # カメラがオープンしており、画像が存在する場合
            self.thread_pool.submit(self._perform_ocr)  # OCR処理をスレッドプールで実行

    def _perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):  # カメラがオープンしており、画像が存在する場合
            self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)  # BGR画像をグレースケールに変換
            cv2.imwrite('gray_paddle.jpg', self.img_gray)  # グレースケール画像をファイルに保存
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHEオブジェクトの作成
            cl1 = clahe.apply(self.img_gray)  # CLAHEの適用
            blurred = automatic_gaussian_blur(cl1)  # ガウスぼかしの適用

            kernel = np.ones((3, 3), dtype=np.uint8)  # カーネルの作成
            dilated = cv2.dilate(blurred, kernel, iterations=2)  # 膨張処理の適用
            eroded = cv2.erode(dilated, kernel, iterations=1)  # 収縮処理の適用

            canny_edges = cv2.Canny(eroded, 30, 150)  # Cannyエッジ検出の適用
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # Sobelフィルタ（x方向）の適用
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Sobelフィルタ（y方向）の適用
            sobel_edges = cv2.magnitude(sobel_x, sobel_y)  # Sobelエッジの計算
            sobel_edges = cv2.convertScaleAbs(sobel_edges)  # 絶対値の変換
            edged = np.maximum(canny_edges, sobel_edges)  # CannyエッジとSobelエッジの最大値を計算
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭の検出
            image_with_contours = cv2.drawContours(self.img_gray.copy(), contours, -1, (0, 255, 0), 2)  # 輪郭を画像に描画

            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # スケールのリスト
            all_results = []  # OCR結果を格納するリスト

            for scale in scales:  # 各スケールに対して処理
                scaled_img = cv2.resize(image_with_contours.copy(), None, fx=scale, fy=scale)  # 画像のリサイズ
                if scaled_img is not None:

                    result_en = self.ocr_en.ocr(scaled_img)  # OCR処理の実行

                    if result_en is not None:  # OCR結果が存在する場合
                        for line in result_en:  # 各ラインに対して処理
                            if line is not None:
                                for word_info in line:  # 各単語に対して処理
                                    word = word_info[1][0]  # 単語の取得
                                    similarity_score = self.calculate_similarity(word)  # 単語の類似度計算
                                    confidence = word_info[1][1]  # OCRの信頼度
                                    if confidence > 0.85 and similarity_score >= 0.85:  # 信頼度と類似度が閾値を超えた場合
                                        rect_color = (0, 255, 0)  # 長方形の色
                                        coordinates = word_info[0]  # 単語の座標
                                        x_min = float(min(pt[0] for pt in coordinates)) / scale  # 最小x座標
                                        y_min = float(min(pt[1] for pt in coordinates)) / scale  # 最小y座標
                                        x_max = float(max(pt[0] for pt in coordinates)) / scale  # 最大x座標
                                        y_max = float(max(pt[1] for pt in coordinates)) / scale  # 最大y座標
                                        all_results.append(
                                            (word, [x_min, y_min, x_max, y_max], confidence, rect_color))  # 結果をリストに追加
                                    # else:
                                    #     rect_color = (0, 0, 255)
                                    #     coordinates = word_info[0]
                                    #     x_min = float(min(pt[0] for pt in coordinates)) / scale
                                    #     y_min = float(min(pt[1] for pt in coordinates)) / scale
                                    #     x_max = float(max(pt[0] for pt in coordinates)) / scale
                                    #     y_max = float(max(pt[1] for pt in coordinates)) / scale
                                    #     all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))

            all_results.sort(key=lambda x: -x[2])  # 信頼度でソート

            keep = [1] * len(all_results)  # 結果を保持するためのリスト
            for i in range(len(all_results)):  # 非最大抑制の適用
                if keep[i] == 0:
                    continue
                _, box1, _, _ = all_results[i]
                for j in range(i + 1, len(all_results)):
                    if keep[j] == 0:
                        continue
                    _, box2, _, _ = all_results[j]
                    if calculate_iou(box1, box2) > 0.25:  # IOUが閾値を超えた場合
                        keep[j] = 0  # 結果を保持しない

            result_img = self.img_bgr.copy()  # 結果画像のコピー

            for k, (word, box, _, rect_color) in enumerate(all_results):  # 結果を画像に描画
                if keep[k] == 1:
                    x_min, y_min, x_max, y_max = map(int, box)  # 座標を整数に変換
                    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color, thickness=2)  # 長方形を描画
                    result_img = draw_text(result_img, word, (x_min, y_min - 5))  # テキストを描画

            cv2.imwrite('Result_paddle.jpg', result_img)  # 結果画像をファイルに保存

            import matplotlib.pyplot as plt  # matplotlibのインポート

            plt.figure("Result")  # 新しい図を作成
            if result_img is None:  # 結果画像が存在しない場合
                print("The result image is None, no image to display")
            else:  # 画像が存在する場合
                rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # RGBに変換
                print("Type of the image array: ", type(rgb_img))  # Image array type
                print("Shape of the image array: ", rgb_img.shape)  # Image array shape
                try:  # Try to display image
                    plt.imshow(rgb_img)  # 画像の表示
                except Exception as e:  # Exception handling if imshow fails
                    print(f"An error occurred when displaying the image: {e}")
            plt.axis('off')  # 軸を非表示に設定
            plt.show()  # 画像の表示

            # ocr_result_en = '\n'.join([word_info[1][0] for line in result_en for word_info in line])  # OCR結果の連結（コメントアウトされている）
            # self.result_label.config(text=ocr_result_en)  # OCR結果をラベルに表示（コメントアウトされている）

    def close(self):
        if self.vid and self.vid.isOpened():  # ビデオキャプチャがオープンしている場合
            self.vid.release()  # ビデオキャプチャを解放
        cv2.destroyAllWindows()  # OpenCVの全ウィンドウを閉じる
        self.thread_pool.shutdown(wait=True)
        self.window.quit()  # Tkinterウィンドウを終了

    def calculate_similarity(self, word):
        for accepted_word, possible_words in ocr_to_accepted_words.items():  # 受け入れ可能な単語リストを確認
            if word in possible_words:
                return 1.0  # 単語が受け入れ可能な単語リストに存在する場合、類似度は1.0
        max_sim = 0  # 最大類似度の初期化
        for accepted_word in ocr_to_accepted_words.keys():  # すべての受け入れ可能な単語と比較
            token1 = nlp(word)  # 単語をSpaCyトークンに変換
            token2 = nlp(accepted_word)  # 受け入れ可能な単語をSpaCyトークンに変換
            spacy_sim = token1.similarity(token2)  # SpaCyによる類似度計算

            fuzzy_sim = fuzz.ratio(word.lower(), accepted_word.lower()) / 100.0  # FuzzyWuzzyによる類似度計算
            sim = max(spacy_sim, fuzzy_sim)  # 最大類似度を選択

            if sim > max_sim:  # 最大類似度を更新
                max_sim = sim
        return max_sim  # 最大類似度を返す

    def process_ocr(self, image_with_contours):
        if image_with_contours is not None:
            result_en = self.ocr_en.ocr(image_with_contours)  # OCR処理の実行
            return result_en

    def update(self):
        if self.camera_open:  # カメラがオープンしている場合
            ret, frame = self.vid.read()  # フレームの読み取り
            if ret:  # フレームが正常に読み取られた場合
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # RGBからBGRに変換

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR画像をグレースケールに変換
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHEオブジェクトの作成
                cl1 = clahe.apply(gray)  # CLAHEの適用
                blurred = automatic_gaussian_blur(cl1)  # ガウスぼかしの適用

                kernel = np.ones((3, 3), dtype=np.uint8)  # カーネルの作成
                dilated = cv2.dilate(blurred, kernel, iterations=2)  # 膨張処理の適用
                eroded = cv2.erode(dilated, kernel, iterations=1)  # 衰退処理の適用

                canny_edges = cv2.Canny(eroded, 30, 150)  # Cannyエッジ検出の実行
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # Sobelフィルタ（x方向）の適用
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Sobelフィルタ（y方向）の適用
                sobel_edges = cv2.magnitude(sobel_x, sobel_y)  # Sobelエッジの計算
                sobel_edges = cv2.convertScaleAbs(sobel_edges)  # 絶対値の変換
                edged = np.maximum(canny_edges, sobel_edges)  # CannyエッジとSobelエッジの最大値を計算
                contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭の検出
                image_with_contours = cv2.drawContours(gray.copy(), contours, -1, (0, 255, 0), 2)  # 輪郭を画像に描画

                with ThreadPoolExecutor() as executor:  # スレッドプールの作成
                    future = executor.submit(self.process_ocr, image_with_contours)  # OCR処理をスレッドプールで実行
                    result_en = future.result()  # 結果の取得

                result_img = img_bgr.copy()  # 結果画像のコピー

                if result_en is not None:  # OCR結果が存在する場合
                    all_results = []  # OCR結果を格納するリスト
                    for line in result_en:  # 各ラインに対して処理
                        if line is not None:
                            for word_info in line:  # 各単語に対して処理
                                word = word_info[1][0]  # 単語の取得
                                similarity_score = self.calculate_similarity(word)  # 単語の類似度計算
                                confidence = word_info[1][1]  # OCRの信頼度
                                if confidence > 0.85 and similarity_score >= 0.85:  # 信頼度と類似度が閾値を超えた場合
                                    coordinates = word_info[0]  # 単語の座標
                                    x_min = int(min(pt[0] for pt in coordinates))  # 最小x座標
                                    y_min = int(min(pt[1] for pt in coordinates))  # 最小y座標
                                    x_max = int(max(pt[0] for pt in coordinates))  # 最大x座標
                                    y_max = int(max(pt[1] for pt in coordinates))  # 最大y座標
                                    rect_color = (0, 255, 0)  # 長方形の色
                                    all_results.append(
                                        (word, [x_min, y_min, x_max, y_max], confidence, rect_color))  # 結果をリストに追加
                                # else:
                                #     rect_color = (0, 0, 255)
                                #     coordinates = word_info[0]
                                #     x_min = int(min(pt[0] for pt in coordinates))
                                #     y_min = int(min(pt[1] for pt in coordinates))
                                #     x_max = int(max(pt[0] for pt in coordinates))
                                #     y_max = int(max(pt[1] for pt in coordinates))
                                #     all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))

                    # Apply Non-Maximum Suppression (NMS) based on IOU
                    keep = [1] * len(all_results)  # 結果を保持するためのリスト
                    for i in range(len(all_results)):  # 非最大抑制の適用
                        if keep[i] == 0:
                            continue
                        _, box1, _, _ = all_results[i]
                        for j in range(i + 1, len(all_results)):
                            if keep[j] == 0:
                                continue
                            _, box2, _, _ = all_results[j]
                            if calculate_iou(box1, box2) > 0.25:  # IOUが閾値を超えた場合
                                keep[j] = 0  # 結果を保持しない

                    for k, (word, box, _, rect_color) in enumerate(all_results):  # 結果を画像に描画
                        if keep[k] == 1:
                            x_min, y_min, x_max, y_max = box  # 座標を取得
                            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color, thickness=2)  # 長方形を描画
                            result_img = draw_text(result_img, word, (x_min, y_min - 5))  # テキストを描画

                    self.photo = ImageTk.PhotoImage(
                        image=Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))  # 画像をTkinter形式に変換
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # Tkinterキャンバスに画像を描画
                    # self.result_label.config(text=ocr_result_en)  # OCR結果をラベルに表示（コメントアウトされている）

        self.window.after(self.delay, self.update)  # 指定した遅延時間後にupdateメソッドを再実行


if __name__ == "__main__":
    try:
        root = tk.Tk()  # Tkinterウィンドウの作成
        app = App(root, "Tkinter and OpenCV")  # アプリケーションのインスタンス化
        root.mainloop()  # Tkinterのメインループを開始
    except KeyboardInterrupt:  # キーボード割り込みが発生した場合
        if app.vid is not None and app.vid.isOpened():  # ビデオキャプチャがオープンしている場合
            app.vid.release()  # ビデオキャプチャを解放
            cv2.destroyAllWindows()  # OpenCVの全ウィンドウを閉じる
            root.quit()  # Tkinterウィンドウを終了
