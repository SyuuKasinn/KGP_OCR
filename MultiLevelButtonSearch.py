import tkinter as tk
from acceptedWords import ocr_to_accepted_words

# Constants
BUTTON_PADDING = 5
BACK_BUTTON_TEXT = "戻る"
RESULT_LABEL_FONT = ('Arial', 14)
SELECTED_TEXT = "あなたが選んだ: "


class MultiLevelButtonSearchApp:
    def __init__(self, root, update_callback):
        self.root = root
        self.update_callback = update_callback
        self.ocr_to_accepted_words = ocr_to_accepted_words
        self.data = self.initialize_data()

        self.current_dict = self.data
        self.path = []

        # Frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.result_label = tk.Label(self.root, text="", font=RESULT_LABEL_FONT)
        self.result_label.grid(row=1, column=0, pady=10, sticky="w")

        self.all_buttons = self.create_all_buttons()
        self.populate_buttons(self.data)

    def initialize_data(self):
        return {
            "カテゴリー": {
                "ラミックス": {key: {key: self.ocr_to_accepted_words[key]} for key in [
                    "ラミックス", "ラミックス2", "ラミックス静岡", "ラミックス群馬",
                    "ラミックス湘南", "ラミックス厚木１", "ラミックス厚木２",
                    "ラミックス横浜２", "ラミックス川崎"]},
                "アブロード": {key: {key: self.ocr_to_accepted_words[key]} for key in [
                    "アブロード横浜", "アブロード杉並府中", "アブロード銀座",
                    "アブロード多摩", "アブロード北東京", "アブロード新宿",
                    "アブロード戸田", "アブロード北東京新宿", "アブロード千葉"]},
                "ムロオ": {key: {key: self.ocr_to_accepted_words[key]} for key in [
                    "ムロオ共配", "ムロオ北東京２"]},
                "柏倉庫(引取り)": {"柏倉庫(引取り)": self.ocr_to_accepted_words['柏倉庫(引取り)']},
                "来　社": {"来　社": self.ocr_to_accepted_words['来　社']},
                "西濃": {"西濃": self.ocr_to_accepted_words['西濃']}
            }
        }

    def create_all_buttons(self):
        buttons = {}

        def add_buttons(current_dict, path):
            for key in current_dict:
                if isinstance(current_dict[key], dict):
                    # Create a button for this key
                    button = tk.Button(self.button_frame, text=key, width=20, height=2,
                                       command=lambda k=key: self.process_click(k))
                    buttons[(tuple(path), key)] = button
                    # Recurse into the next level
                    add_buttons(current_dict[key], path + [key])
                else:
                    # This is a leaf node
                    button = tk.Button(self.button_frame, text=key, width=20, height=2,
                                       command=lambda k=key: self.process_click(k))
                    buttons[(tuple(path), key)] = button

        add_buttons(self.data, [])
        return buttons

    def populate_buttons(self, dictionary):
        # Clear previous widgets from the grid
        for widget in self.button_frame.winfo_children():
            widget.grid_forget()

        # Grid configuration
        self.button_frame.columnconfigure(0, weight=1)  # Make the first column expandable
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.columnconfigure(2, weight=1)

        # Add back button if needed
        if self.path:
            back_button = tk.Button(self.button_frame, text=BACK_BUTTON_TEXT, width=10, height=1, bg="red", fg="white",
                                    command=self.back)
            back_button.grid(row=0, column=0, padx=BUTTON_PADDING, pady=BUTTON_PADDING, sticky="w")

        # Add buttons for current dictionary
        row = 1
        column = 0
        for key in dictionary.keys():
            if (tuple(self.path), key) in self.all_buttons:
                button = self.all_buttons[(tuple(self.path), key)]
                button.grid(row=row, column=column, padx=BUTTON_PADDING, pady=BUTTON_PADDING, sticky="ew")

                # Move to the next column
                column += 1
                if column > 2:  # Wrap to the next row after 3 columns
                    column = 0
                    row += 1

    def process_click(self, key):
        self.path.append(key)

        if isinstance(self.current_dict[key], dict):
            self.current_dict = self.current_dict[key]
            self.populate_buttons(self.current_dict)
        else:
            self.result_label.config(text=f"{SELECTED_TEXT}{self.path[-1]}")
            self.update_callback(self.path[-1])

    def back(self):
        if not self.path:
            return
        self.path.pop()
        self.current_dict = self.data
        for key in self.path:
            self.current_dict = self.current_dict.get(key, {})
            if not isinstance(self.current_dict, dict):
                return

        self.populate_buttons(self.current_dict)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLevelButtonSearchApp(root, lambda x: print(f"Selected: {x}"))
    root.mainloop()
