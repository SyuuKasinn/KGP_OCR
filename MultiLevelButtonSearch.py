import tkinter as tk
from acceptedWords import ocr_to_accepted_words


class MultiLevelButtonSearchApp:
    def __init__(self, root, update_callback):
        self.root = root
        # self.root.title("多级按钮点击式搜索系统")

        self.ocr_to_accepted_words = ocr_to_accepted_words

        self.data = {
            "カテゴリー": {
                "ラミックス": {
                    "ラミックス": {'ラミックス': self.ocr_to_accepted_words['ラミックス']},
                    "ラミックス2": {'ラミックス2': self.ocr_to_accepted_words['ラミックス2']},
                    "ラミックス静岡": {'ラミックス静岡': self.ocr_to_accepted_words['ラミックス静岡']},
                    "ラミックス群馬": {'ラミックス群馬': self.ocr_to_accepted_words['ラミックス群馬']},
                    "ラミックス湘南": {'ラミックス湘南': self.ocr_to_accepted_words['ラミックス湘南']},
                    "ラミックス厚木１": {'ラミックス厚木１': self.ocr_to_accepted_words['ラミックス厚木１']},
                    "ラミックス厚木２": {'ラミックス厚木２': self.ocr_to_accepted_words['ラミックス厚木２']},
                    "ラミックス横浜２": {'ラミックス横浜２': self.ocr_to_accepted_words['ラミックス横浜２']},
                    "ラミックス川崎": {'ラミックス川崎': self.ocr_to_accepted_words['ラミックス川崎']},

                },
                "アブロード": {
                    "アブロード横浜": {'アブロード横浜': self.ocr_to_accepted_words['アブロード横浜']},
                    "アブロード杉並府中": {'アブロード杉並府中': self.ocr_to_accepted_words['アブロード杉並府中']},
                    "アブロード銀座": {'アブロード銀座': self.ocr_to_accepted_words['アブロード銀座']},
                    "アブロード多摩": {'アブロード多摩': self.ocr_to_accepted_words['アブロード多摩']},
                    "アブロード北東京": {'アブロード北東京': self.ocr_to_accepted_words['アブロード北東京']},
                    "アブロード新宿": {'アブロード新宿': self.ocr_to_accepted_words['アブロード新宿']},
                    "アブロード戸田": {'アブロード戸田': self.ocr_to_accepted_words['アブロード戸田']},
                    "アブロード北東京新宿": {
                        'アブロード北東京新宿': self.ocr_to_accepted_words['アブロード北東京新宿']},
                    "アブロード千葉": {'アブロード千葉': self.ocr_to_accepted_words['アブロード千葉']},
                },
                "ムロオ": {
                    "ムロオ共配": {'ムロオ共配': self.ocr_to_accepted_words['ムロオ共配']},
                    "ムロオ北東京２": {'ムロオ北東京２': self.ocr_to_accepted_words['ムロオ北東京２']},
                },
                "柏倉庫(引取り)": {
                    "柏倉庫(引取り)": {'柏倉庫(引取り)': self.ocr_to_accepted_words['柏倉庫(引取り)']},
                },
                "来　社": {
                    "来　社": {'来　社': self.ocr_to_accepted_words['来　社']},
                },
                "西濃": {
                    "西濃": {'西濃': self.ocr_to_accepted_words['西濃']},
                }
            }
        }

        self.current_dict = self.data
        self.path = []

        # Frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=20, padx=20)

        self.result_label = tk.Label(self.root, text="", font=('Arial', 14))
        self.result_label.pack(pady=10)

        # Populate category buttons
        self.populate_buttons(self.data)

        self.ocr_label = []
        self.update_callback = update_callback

    def populate_buttons(self, dictionary):
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        if self.path:
            back_button = tk.Button(self.button_frame, text="戻る", bg="red", fg="white", command=self.back)
            back_button.pack(side=tk.LEFT, padx=20, pady=20)

        for key in dictionary.keys():
            button = tk.Button(self.button_frame, text=key, command=lambda k=key: self.process_click(k))
            button.pack(side=tk.LEFT, padx=5, pady=5)

    def process_click(self, key):
        self.path.append(key)
        if isinstance(self.current_dict[key], dict):
            # If the clicked item is a dictionary, make it the current dictionary
            # and create new buttons
            self.current_dict = self.current_dict[key]
            self.populate_buttons(self.current_dict)
        elif isinstance(self.current_dict[key], set):
            # If the clicked item is a set, it is a final item.
            # Show the selected item (the last item in the path)
            self.result_label.config(text=f"あなたが選んだ: {self.path[-1]}")
            self.ocr_label = self.path[-1]
            self.update_callback(self.ocr_label)
            # Optionally, if you want to show all the keys under that item, you can uncomment the following line
            # self.result_label.config(text=f"您选择了: {self.path[-1]} | Keys: {', '.join(self.current_dict[key])}")
        else:
            # This is not expected to happen with the current structure of data,
            # but if it does, we can just display the last item in the path
            self.result_label.config(text=f"あなたが選んだ: {self.path[-1]}")
            self.ocr_label = self.path[-1]
            self.update_callback(self.ocr_label)

    def back(self):
        # Get rid of the current level's name
        self.path.pop()
        # Go back to parent level dictionary
        self.current_dict = self.data
        for key in self.path:
            self.current_dict = self.current_dict[key]

        self.populate_buttons(self.current_dict)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLevelButtonSearchApp(root)
    root.mainloop()
