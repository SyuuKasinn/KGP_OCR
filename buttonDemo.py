import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from paddleOCR_Video_spacy_20240807_05 import App
from MultiLevelButtonSearch20240807_01 import MultiLevelButtonSearchApp


class MainApp:
    def __init__(self, root):
        self.root = root
        self.custom_font = tkfont.Font(family="Helvetica", size=12)  # カスタムフォント

        # ttkthemesを使用する場合は、以下の行のコメントを外し、テーマを選択します
        # self.root = ttkthemes.ThemedTk()
        # self.root.get_themes()
        # self.root.set_theme("plastik")  # テーマの例

        self.root.title("組み合わせGUIアプリケーション")

        # 主なフレームを作成し、グリッドレイアウトを使用
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 一貫した外観のためのスタイルを定義
        self.style = ttk.Style()
        self.style.configure('TFrame')

        # App1とApp2のフレームを作成して配置
        self.create_app1_frame()
        self.create_app2_frame()

    def create_app1_frame(self):
        self.app1_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN)
        self.app1_frame.grid(row=0, column=0, sticky='nsew')

        # App1を初期化して配置
        self.app1 = App(self.app1_frame, "App1のタイトル")
        self.app1_frame.grid_rowconfigure(0, weight=1)
        self.app1_frame.grid_columnconfigure(0, weight=1)

    def create_app2_frame(self):
        self.app2_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.SUNKEN)
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
        if hasattr(self.app1, 'ocr_label'):
            self.app1.update_ocr_label(ocr_label)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
