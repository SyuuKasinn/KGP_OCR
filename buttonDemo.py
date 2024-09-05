import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from paddleOCR_Video_spacy_20240802_04 import App
from buttonDemo20240802 import MultiLevelButtonSearchApp


class MainApp:
    def __init__(self, root):
        self.root = root
        customFont = tkfont.Font(family="Helvetica", size=12)  # Customize font

        # Using ttkthemes
        # self.root = ttkthemes.ThemedTk()  # Uncomment if using ttkthemes
        # self.root.get_themes()
        # self.root.set_theme("plastik")  # Uncomment and change to your liking if using ttkthemes

        self.root.title("Combined GUI Application")

        self.main_frame = tk.Frame(self.root)

        self.main_frame.pack(fill=tk.BOTH,
                             expand=True,
                             padx=10,
                             pady=10)

        # Create and place App1 and App2 frames
        self.app1_frame = tk.Frame(self.main_frame,
                                   bd=5,
                                   relief=tk.SUNKEN)  # Change border, color

        self.app1 = App(self.app1_frame, "Title for App1")
        self.app1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)  # Add padding

        self.app2_frame = tk.Frame(self.main_frame,
                                   bd=5,
                                   relief=tk.SUNKEN)  # Change border, color

        self.app2 = MultiLevelButtonSearchApp(self.app2_frame)
        self.app2_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)  # Add padding


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
