import math
import threading
import tkinter as tk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import spacy
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from paddleocr import PaddleOCR
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
nlp = spacy.load('ja_core_news_lg')

accepted_words = {'秩父の天然水', '父の天然水', '稚父の天然水', '种父の天然水'}


def automatic_gaussian_blur(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = math.floor(gray.shape[1] / 20)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blurred


def draw_text(image, text, position):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font_path = 'M_PLUS_1p/MPLUS1p-Regular.ttf'
    font = ImageFont.truetype(font_path, 20)
    draw.text(position, text, font=font, fill=(0, 0, 255, 0))
    return np.array(img_pil)


def calculate_iou(box1, box2):
    xi1, yi1, xi2, yi2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = None
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang='japan')
        self.result_label = tk.Label(window, text="OCR Result")
        self.result_label.pack()
        self.camera_open = False
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.buttons = {
            "Open Camera": self.open_camera,
            "Snapshot": self.take_snapshot,
            "OCR": self.perform_ocr,
            "Exit": self.close
        }

        for button_text, button_command in self.buttons.items():
            tk.Button(window, text=button_text, width=20, command=button_command).pack()

        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.delay = 50
        self.update()

    def open_camera(self):
        if not self.camera_open:
            self.vid = cv2.VideoCapture(self.video_source)
            if self.vid.isOpened():
                self.camera_open = True
                self.canvas.config(width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                   height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def take_snapshot(self):
        if self.camera_open:
            ret, frame = self.vid.read()
            if ret:
                self.img_bgr = frame.copy()
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                import matplotlib.pyplot as plt
                plt.figure("Snapshot")
                plt.imshow(self.img_rgb)
                plt.axis('off')
                plt.show()

    def perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):
            self.thread_pool.submit(self._perform_ocr)
            self.perform_ocr_thread = threading.Thread(target=self._perform_ocr)
            self.perform_ocr_thread.start()

    def _perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):
            self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gray_paddle.jpg', self.img_gray)

            blurred = automatic_gaussian_blur(self.img_gray)
            edged = cv2.Canny(blurred, 30, 150)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = cv2.drawContours(self.img_gray.copy(), contours, -1, (0, 255, 0), 2)

            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            all_results = []

            for scale in scales:
                scaled_img = cv2.resize(image_with_contours.copy(), None, fx=scale, fy=scale)
                result_en = self.ocr_en.ocr(scaled_img)

                if result_en is not None:
                    for line in result_en:
                        if line is not None:
                            for word_info in line:
                                word = word_info[1][0]
                                similarity_score = self.calculate_similarity(word)
                                confidence = word_info[1][1]
                                if confidence > 0.85 and similarity_score >= 0.85:
                                    rect_color = (0, 255, 0)
                                    coordinates = word_info[0]
                                    x_min = float(min(pt[0] for pt in coordinates)) / scale
                                    y_min = float(min(pt[1] for pt in coordinates)) / scale
                                    x_max = float(max(pt[0] for pt in coordinates)) / scale
                                    y_max = float(max(pt[1] for pt in coordinates)) / scale
                                    all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))
                                else:
                                    rect_color = (0, 0, 255)
                                    coordinates = word_info[0]
                                    x_min = float(min(pt[0] for pt in coordinates)) / scale
                                    y_min = float(min(pt[1] for pt in coordinates)) / scale
                                    x_max = float(max(pt[0] for pt in coordinates)) / scale
                                    y_max = float(max(pt[1] for pt in coordinates)) / scale
                                    all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))

            all_results.sort(key=lambda x: -x[2])

            keep = [1] * len(all_results)
            for i in range(len(all_results)):
                if keep[i] == 0:
                    continue
                _, box1, _, _ = all_results[i]
                for j in range(i + 1, len(all_results)):
                    if keep[j] == 0:
                        continue
                    _, box2, _, _ = all_results[j]
                    if calculate_iou(box1, box2) > 0.25:
                        keep[j] = 0

            result_img = self.img_bgr.copy()

            for k, (word, box, _, rect_color) in enumerate(all_results):
                if keep[k] == 1:
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color, thickness=2)
                    result_img = draw_text(result_img, word, (x_min, y_min - 5))

            cv2.imwrite('Result_paddle.jpg', result_img)

            import matplotlib.pyplot as plt
            plt.figure("Result")
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            # ocr_result_en = '\n'.join([word_info[1][0] for line in result_en for word_info in line])
            # self.result_label.config(text=ocr_result_en)

    def close(self):
        if self.vid and self.vid.isOpened():
            self.vid.release()
        cv2.destroyAllWindows()
        self.window.quit()

    def draw_text(image, text, position):
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype('path/to/mplus-1p-regular.ttf', 20)
        draw.text(position, text, font=font, fill=(0, 0, 255, 0))
        return np.array(img_pil)

    def calculate_similarity(self, word):
        if word in accepted_words:
            return 1.0
        max_sim = 0
        for accepted_word in accepted_words:
            token1 = nlp(word)
            token2 = nlp(accepted_word)
            sim = token1.similarity(token2)
            if sim > max_sim:
                max_sim = sim
        return max_sim

    def update(self):
        if self.camera_open:
            ret, frame = self.vid.read()
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = automatic_gaussian_blur(gray)
                edged = cv2.Canny(blurred, 30, 150)
                contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image_with_contours = cv2.drawContours(gray.copy(), contours, -1, (0, 255, 0), 2)
                result_en = self.ocr_en.ocr(image_with_contours)
                result_img = img_bgr.copy()

                if result_en is not None:
                    ocr_result_en = ''
                    for line in result_en:
                        if line is not None:
                            for word_info in line:
                                word = word_info[1][0]
                                similarity_score = self.calculate_similarity(word)
                                if similarity_score >= 0.85:
                                    rect_color = (0, 255, 0)
                                else:
                                    rect_color = (0, 0, 255)
                                ocr_result_en += word + '\n'
                                confidence = word_info[1][1]
                                if confidence > 0.85:
                                    coordinates = word_info[0]
                                    x_min = int(min(pt[0] for pt in coordinates))
                                    y_min = int(min(pt[1] for pt in coordinates))
                                    x_max = int(max(pt[0] for pt in coordinates))
                                    y_max = int(max(pt[1] for pt in coordinates))
                                    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color, thickness=2
                                                  )
                                    result_img = draw_text(result_img, word, (x_min, y_min - 5))

                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    # self.result_label.config(text=ocr_result_en)

        self.window.after(self.delay, self.update)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root, "Tkinter and OpenCV")
        root.mainloop()
    except KeyboardInterrupt:
        if app.vid is not None and app.vid.isOpened():
            app.vid.release()
            cv2.destroyAllWindows()
            root.quit()
