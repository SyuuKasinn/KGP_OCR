import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from paddleOCR_Video_spacy import App
from MultiLevelButtonSearch import MultiLevelButtonSearchApp
import queue

class MainApp:
    def __init__(self, root):
        self.root = root
        self.custom_font = tkfont.Font(family="Helvetica", size=12)

        self.root.title("統合型GUIアプリケーション")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        # メインフレームの作成
        self.main_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # スタイルの設定
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', font=self.custom_font, background='#e0e0e0')

        # アプリフレームの作成
        self.create_app1_frame()
        self.create_app2_frame()

        # スレッド間通信のためのキュー
        self.queue = queue.Queue()
        self.root.after(10, self.check_queue)

    def create_app1_frame(self):
        self.app1_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN)
        self.app1_frame.grid(row=0, column=0, sticky='nsew')

        # App1の初期化
        try:
            self.app1 = App(self.app1_frame, "App1のタイトル")
        except Exception as e:
            print(f"App1の初期化中にエラーが発生しました: {e}")
            self.app1 = None

        self.app1_frame.grid_rowconfigure(0, weight=1)
        self.app1_frame.grid_columnconfigure(0, weight=1)

    def create_app2_frame(self):
        self.app2_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN)
        self.app2_frame.grid(row=0, column=1, sticky='nsew')

        # App2の初期化
        try:
            self.app2 = MultiLevelButtonSearchApp(self.app2_frame, self.update_app1_ocr_label)
        except Exception as e:
            print(f"App2の初期化中にエラーが発生しました: {e}")
            self.app2 = None

        self.app2_frame.grid_rowconfigure(0, weight=1)
        self.app2_frame.grid_columnconfigure(0, weight=1)

        # フレームが利用可能なスペースを広げて埋めるように設定
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

    def update_app1_ocr_label(self, ocr_label):
        if self.app1:
            threading.Thread(target=self._update_label, args=(ocr_label,), daemon=True).start()
        else:
            print("App1が初期化されていないため、OCRラベルを更新できません。")

    def _update_label(self, ocr_label):
        self.queue.put(ocr_label)

    def check_queue(self):
        while not self.queue.empty():
            ocr_label = self.queue.get()
            if hasattr(self.app1, 'update_ocr_label'):
                try:
                    self.app1.update_ocr_label(ocr_label)
                except Exception as e:
                    print(f"App1のOCRラベルの更新中にエラーが発生しました: {e}")
        self.root.after(10, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
