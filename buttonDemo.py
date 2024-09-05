import tkinter as tk
from acceptedWords import ocr_to_accepted_words

class MultiLevelButtonSearchApp:
    def __init__(self, root):
        self.root = root
        #self.root.title("多级按钮点击式搜索系统")

        self.ocr_to_accepted_words =ocr_to_accepted_words

        self.data = {
            "Foods": {
                "Fruits": {
                    "Citrus": {"Orange": None, "Lemon": None},
                    "Non-Citrus": {"Apple": None, "Banana": None},
                    "秩父": self.ocr_to_accepted_words
                },
                "Vegetables": {"Carrot": None, "Broccoli": None}
            }
        }

        self.current_dict = self.data
        self.path = []

        # Frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10, padx=10)

        self.result_label = tk.Label(self.root, text="", font=('Arial', 14))
        self.result_label.pack(pady=10)

        # Populate category buttons
        self.populate_buttons(self.data)

    def populate_buttons(self, dictionary):
        for widget in self.button_frame.winfo_children():
            widget.destroy()

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
            self.result_label.config(text=f"您选择了: {self.path[-1]}")
            # Optionally, if you want to show all the keys under that item, you can uncomment the following line
            # self.result_label.config(text=f"您选择了: {self.path[-1]} | Keys: {', '.join(self.current_dict[key])}")
        else:
            # This is not expected to happen with the current structure of data,
            # but if it does, we can just display the last item in the path
            self.result_label.config(text=f"您选择了: {self.path[-1]}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLevelButtonSearchApp(root)
    root.mainloop()
