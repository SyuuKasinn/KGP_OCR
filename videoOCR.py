import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = None
        self.ocr_japan = PaddleOCR(use_angle_cls=True, lang='japan')
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang='en')
        self.result_label = tk.Label(window)
        self.result_label.pack()

        self.camera_open = False
        self.buttons = {
            "Open Camera": self.open_camera,
            "Snapshot": self.snapshot,
            "OCR": self.perform_ocr,
            "Exit": self.close
        }
        for button_text, button_command in self.buttons.items():
            tk.Button(window, text=button_text, width=20, command=button_command).pack()

        self.window.mainloop()

    def open_camera(self):
        if not self.camera_open:
            self.vid = cv2.VideoCapture(self.video_source)
            self.canvas = tk.Canvas(self.window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.canvas.pack()
            self.delay = 15
            self.update()

    def snapshot(self):
        ret, self.img_bgr = self.vid.read()

        if ret:
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def perform_ocr(self):
        self.vid.release()
        self.camera_open = False

        if self.img_bgr is not None:
            self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gray_paddle.jpg', self.img_gray)

            result_en = self.ocr_en.ocr(self.img_bgr)
            # result_japan = self.ocr_japan.ocr(self.img_bgr)
            result_img = self.img_bgr.copy()

            for line in result_en:
                for word_info in line:
                    word = word_info[1][0]
                    coordinates = word_info[0]
                    x_min = int(min(pt[0] for pt in coordinates))
                    y_min = int(min(pt[1] for pt in coordinates))
                    x_max = int(max(pt[0] for pt in coordinates))
                    y_max = int(max(pt[1] for pt in coordinates))
                    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(result_img, word, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # for line in result_japan:
            #     for word_info in line:
            #         word = word_info[1][0]
            #         coordinates = word_info[0]
            #         x_min = int(min(pt[0] for pt in coordinates))
            #         y_min = int(min(pt[1] for pt in coordinates))
            #         x_max = int(max(pt[0] for pt in coordinates))
            #         y_max = int(max(pt[1] for pt in coordinates))
            #         cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #         cv2.putText(result_img, word, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imwrite('Result_paddle.jpg', result_img)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            # Extract recognized text
            ocr_result_en = '\n'.join([word_info[1][0] for line in result_en for word_info in line])
            #  ocr_result_japan = '\n'.join([word_info[1][0] for line in result_japan for word_info in line])

            self.result_label.config(text=ocr_result_en)

    def close(self):
        if self.vid and self.vid.isOpened():
            self.vid.release()
        self.window.quit()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


App(tk.Tk(), "Tkinter and OpenCV")
