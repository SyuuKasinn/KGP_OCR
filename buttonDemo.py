import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from paddleOCR_Video_spacy_20240808_05 import App
from MultiLevelButtonSearch20240808_07 import MultiLevelButtonSearchApp
import queue


class MainApp:
    def __init__(self, root):
        self.root = root
        self.custom_font = tkfont.Font(family="Helvetica", size=12)  # カスタムフォント

        self.root.title("統合型GUIアプリケーション")
        self.root.geometry("1000x600")  # メインウィンドウのサイズを設定
        self.root.resizable(False, False)  # リサイズ不可

        # 主なフレームを作成し、グリッドレイアウトを使用
        self.main_frame = tk.Frame(self.root, width=1200, height=800, bg='#f0f0f0')
        self.main_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        # 一貫した外観のためのスタイルを定義
        self.style = ttk.Style()
        self.style.configure('TFrame')  # 框架的浅灰色背景
        self.style.configure('TLabel', font=self.custom_font, background='#e0e0e0')

        # App1とApp2のフレームを作成して配置
        self.create_app1_frame()
        self.create_app2_frame()

        self.queue = queue.Queue()
        self.root.after(10, self.check_queue)

    def create_app1_frame(self):
        self.app1_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN, width=380, height=580)
        self.app1_frame.grid(row=0, column=0, sticky='nsew')

        # App1を初期化して配置
        self.app1 = App(self.app1_frame, "App1のタイトル")
        self.app1_frame.grid_rowconfigure(0, weight=1)
        self.app1_frame.grid_columnconfigure(0, weight=1)

    def create_app2_frame(self):
        self.app2_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN, width=380, height=580)
        self.app2_frame.grid(row=0, column=1, sticky='nsew')

        # App2を初期化して配置
        self.app2 = MultiLevelButtonSearchApp(self.app2_frame, self.update_app1_ocr_label)
        self.app2_frame.grid_rowconfigure(0, weight=1)
        self.app2_frame.grid_columnconfigure(0, weight=1)

        # 列と行が比例して拡張するように設定
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

    def update_app1_ocr_label(self, ocr_label):
        # 使用线程更新UI
        threading.Thread(target=self._update_label, args=(ocr_label,)).start()

    def _update_label(self, ocr_label):
        self.queue.put(ocr_label)

    def check_queue(self):
        while not self.queue.empty():
            ocr_label = self.queue.get()
            if hasattr(self.app1, 'ocr_label'):
                self.app1.update_ocr_label(ocr_label)
        self.root.after(10, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
