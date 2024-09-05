import functools
import tkinter as tk
from acceptedWords import ocr_to_accepted_words

# 定数
BUTTON_PADDING = 5
BACK_BUTTON_TEXT = "戻る"
RESULT_LABEL_FONT = ('Arial', 14)
SELECTED_TEXT = "あなたが選んだ: "
BUTTON_WIDTH = 20
BUTTON_HEIGHT = 2


def initialize_data():
    """
    ボタン階層のデータ構造を初期化します。

    戻り値:
        dict: ボタンのカテゴリーとそのアイテムを表すネストされた辞書。
    """
    return {
        "カテゴリー": {
            "ラミックス": {key: {key: ocr_to_accepted_words[key]} for key in [
                "ラミックス", "ラミックス2", "ラミックス静岡", "ラミックス群馬",
                "ラミックス湘南", "ラミックス厚木１", "ラミックス厚木２",
                "ラミックス横浜２", "ラミックス川崎"]},
            "アブロード": {key: {key: ocr_to_accepted_words[key]} for key in [
                "アブロード横浜", "アブロード杉並府中", "アブロード銀座",
                "アブロード多摩", "アブロード北東京", "アブロード新宿",
                "アブロード戸田", "アブロード北東京新宿", "アブロード千葉"]},
            "ムロオ": {key: {key: ocr_to_accepted_words[key]} for key in [
                "ムロオ共配", "ムロオ北東京２"]},
            "柏倉庫(引取り)": {"柏倉庫(引取り)": ocr_to_accepted_words['柏倉庫(引取り)']},
            "来　社": {"来　社": ocr_to_accepted_words['来　社']},
            "西濃": {"西濃": ocr_to_accepted_words['西濃']}
        }
    }


class MultiLevelButtonSearchApp:
    def __init__(self, root, update_callback):
        """
        MultiLevelButtonSearchApp インスタンスを初期化します。

        引数:
            root (tk.Tk): Tkinter のルートウィンドウ。
            update_callback (callable): 選択された値を処理するコールバック関数。
        """
        self.root = root
        self.update_callback = update_callback
        self.data = initialize_data()
        self.current_dict = self.data
        self.path = []

        # UI コンポーネントを作成
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.result_label = tk.Label(self.root, text="", font=RESULT_LABEL_FONT)
        self.result_label.grid(row=1, column=0, pady=10, sticky="w")

        self.all_buttons = self.create_all_buttons()
        self.populate_buttons(self.data)

    def create_button(self, text, command):
        """
        指定されたテキストとコマンドでボタンを作成します。

        引数:
            text (str): ボタンに表示するテキスト。
            command (callable): ボタンがクリックされたときに実行されるコマンド。

        戻り値:
            tk.Button: 作成されたボタン。
        """
        return tk.Button(self.button_frame, text=text, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=command)

    def create_all_buttons(self):
        """
        階層のすべてのレベルのボタンを作成します。

        戻り値:
            dict: (path, key) タプルからボタンへのマッピング辞書。
        """
        buttons = {}

        def add_buttons(current_dict, path):
            for key in current_dict:
                command = functools.partial(self.process_click, key)
                button = self.create_button(key, command)
                buttons[(tuple(path), key)] = button
                if isinstance(current_dict[key], dict):
                    path.append(key)
                    add_buttons(current_dict[key], path)
                    path.pop()

        add_buttons(self.data, [])
        return buttons

    def populate_buttons(self, dictionary):
        """
        現在の辞書レベルに基づいてボタンフレームにボタンを配置します。

        引数:
            dictionary (dict): 現在のレベルを表す辞書。
        """
        # 古いボタンをクリア
        for widget in self.button_frame.winfo_children():
            widget.grid_forget()

        # 列の重み設定
        self.button_frame.columnconfigure([0, 1, 2], weight=1)

        # 戻るボタンを追加する場合
        if self.path:
            back_button = self.create_button(BACK_BUTTON_TEXT, self.back)
            back_button.config(width=10, height=1, bg="red", fg="white")
            back_button.grid(row=0, column=0, padx=BUTTON_PADDING, pady=BUTTON_PADDING, sticky="w")

        # 現在のレベルのボタンを追加
        row, column = 1, 0
        for key in dictionary:
            if (tuple(self.path), key) in self.all_buttons:
                button = self.all_buttons[(tuple(self.path), key)]
                button.grid(row=row, column=column, padx=BUTTON_PADDING, pady=BUTTON_PADDING, sticky="ew")
                column += 1
                if column > 2:
                    column = 0
                    row += 1

    def process_click(self, key):
        """
        ボタンクリックを処理して、現在の辞書と UI を更新します。

        引数:
            key (str): クリックされたボタンに関連付けられたキー。
        """
        try:
            clicked_button = self.all_buttons.get((tuple(self.path), key))
            if clicked_button is not None:
                if not isinstance(self.current_dict[key], dict):
                    clicked_button.config(state=tk.DISABLED)

                self.path.append(key)
                if isinstance(self.current_dict[key], dict):
                    self.current_dict = self.current_dict[key]
                    self.populate_buttons(self.current_dict)
                else:
                    self.result_label.config(
                        text=f"{SELECTED_TEXT}{self.path[-1]}",
                        font=('Arial', 14, 'bold'),
                        fg='black',
                        background='#F5F5DC'
                    )
                    self.update_callback(self.path[-1])

        except KeyError as e:
            print(f"KeyError が発生しました: {e}")
            self.result_label.config(text="選択された項目が無効です。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            self.result_label.config(text="エラーが発生しました。もう一度お試しください。")

    def back(self):
        """
        戻るナビゲーションを処理して、現在の辞書と UI を更新します。
        """
        if not self.path:
            return

        last_key = self.path.pop()
        self.current_dict = self.data
        for key in self.path:
            if key in self.current_dict and isinstance(self.current_dict[key], dict):
                self.current_dict = self.current_dict[key]
            else:
                self.current_dict = self.data
                break

        for (path, button_key), button in self.all_buttons.items():
            if path == tuple(self.path) and button_key == last_key:
                button.config(state=tk.NORMAL)

        self.populate_buttons(self.current_dict)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLevelButtonSearchApp(root, lambda x: print(f"選択された: {x}"))
    root.mainloop()
