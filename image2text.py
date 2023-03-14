import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageGrab
from clip_interrogator import Config, Interrogator
import torch
import random
import re
import base64

# if you want this to cache to a specific folder
# make sure you set TRANSFORMERS_CACHE environment variable
# $env:TRANSFORMERS_CACHE = "./cache_directory/"
# or
# import os
# os.environ["TRANSFORMERS_CACHE"] = "./cache_directory/"

model_name = "ViT-L-14/openai"
selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_word_limit = 77


def count_words(text):
    # remove any non-word characters (e.g. punctuation, special characters)
    text = re.sub(r'[^\w\s]', '', text)
    # split the text into words and count the number of words
    words = re.findall(r'\w+', text)
    return len(words)


class TextExtractorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CLIP-BLIP Image2Text Prompt Tool")
        self.master.geometry("809x403")

        # load model
        self.loading_screen = tk.Toplevel(self.master)
        self.loading_screen.title("Loading Model")
        self.loading_screen.geometry("100x100")
        ttk.Label(self.loading_screen, text="Loading Model...").grid(
            row=0, column=0)
        self.loading_screen.lift()
        self.loading_screen.update()

        self.config = Config(device=selected_device,
                             clip_model_name=model_name, quiet=True)
        self.config.download_cache = True
        self.ci = Interrogator(self.config)

        self.text = "test"

        # create menu bar
        menu_bar = tk.Menu(self.master)

        # create File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_file)
        file_menu.add_command(
            label="Image2Text from clipboard", command=self.on_clipboard)
        file_menu.add_command(
            label="Copy result to clipboard", command=self.on_copy)
        file_menu.add_separator()
        file_menu.add_command(label="Export List", command=self.export_list)
        file_menu.add_command(label="Import List", command=self.import_list)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # set menu bar
        self.master.config(menu=menu_bar)

        # fast mode
        self.fastMode = tk.BooleanVar(value=True)
        self.c_usefast = tk.Checkbutton(
            self.master, text='Use Fast Mode',
            variable=tk.BooleanVar(value=True), onvalue=True, offvalue=False)
        self.c_usefast.grid(row=0, column=0, pady=10, sticky="w")
        self.c_usefast.select()

        # list of prompts
        self.label_list = tk.Label(self.master, text="Prompt List")
        self.label_list.grid(row=1, column=0)
        self.prompt_listbox = tk.Listbox(
            self.master, height=10, selectmode=tk.EXTENDED, width=50)
        self.prompt_listbox.grid(row=2, column=0, pady=10, padx=30)
        self.prompt_listbox.bind(
            "<Double-Button-1>", self.display_selected_prompt)

        # buttons
        self.merge_button = tk.Button(
            self.master, text="Merge Selected Prompts", command=self.merge_prompts)
        self.merge_button.grid(row=3, column=1, pady=10)

        self.clear_button = tk.Button(
            self.master, text="Clear List", command=self.clear_listbox)
        self.clear_button.grid(row=3, column=0)

        # prompt output
        self.label_prompt = tk.Label(self.master, text="Prompt Output")
        self.label_prompt.grid(row=1, column=1)
        self.text_widget = tk.Text(self.master, height=10, width=50)
        self.text_widget.grid(row=2, column=1, pady=10)

        # finished loading
        self.loading_screen.destroy()

    def extract_text_from_image(self, image_path):
        self.image = Image.open(image_path)
        if self.fastMode:
            self.text = self.ci.interrogate_fast(self.image)
        else:
            self.text = self.ci.interrogate(self.image)
        self.prompt_listbox.insert(
            tk.END, self.text, width=100)  # add prompt to listbox
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, self.text)

    def on_clipboard(self):
        self.image = ImageGrab.grabclipboard()
        if self.image is None:
            messagebox.showerror("Error", "No image in clipboard.")
            return
        if self.fastMode:
            self.text = self.ci.interrogate_fast(self.image)
        else:
            self.text = self.ci.interrogate(self.image)
        self.prompt_listbox.insert(tk.END, self.text)  # add prompt to listbox
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, self.text)

    def on_copy(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.text)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        self.extract_text_from_image(file_path)

    def display_selected_prompt(self, event):
        selected_item = self.prompt_listbox.curselection()
        if selected_item:
            prompt_index = selected_item[0]
            selected_prompt = self.prompt_listbox.get(prompt_index)
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(tk.END, selected_prompt)

    def clear_listbox(self):
        self.prompt_listbox.delete(0, tk.END)
        self.text_widget.delete(1.0, tk.END)

    def merge_prompts(self):
        selected_items = self.prompt_listbox.curselection()
        if len(selected_items) < 2:
            messagebox.showerror(
                "Error", "Please select at least 2 prompts to merge")
            return

        prompt_indices = [int(i) for i in selected_items]
        prompt_texts = [self.prompt_listbox.get(i) for i in prompt_indices]
        temp_texts = []
        prompt_starts = []
        for prompt in prompt_texts:
            prompt_parts = prompt.split(", ")
            prompt_starts.append(prompt_parts[0] + ", ")
            prompt_parts = prompt_parts[1:]
            temp_texts.append(', '.join(prompt_parts))
        prompt_texts = temp_texts

        merged_prompt = prompt_starts[random.randint(0, len(prompt_starts))]
        parts = [part.split(", ") for part in prompt_texts]
        max_parts = max([len(part) for part in parts])
        loop_end = False

        for i in range(max_parts):
            for part in prompt_texts:
                parts = part.split(", ")
                random.shuffle(parts)
                if i < len(parts):
                    if random.random() > 1 / len(selected_items):
                        merged = merged_prompt + parts[i] + ", "
                        if count_words(merged) > prompt_word_limit:
                            loop_end = True
                            break
                        merged_prompt = merged
            if loop_end:
                break

        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, merged_prompt)

    def export_list(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, "wb") as f:
                for prompt in self.prompt_listbox.get(0, tk.END):
                    message_bytes = prompt.encode('ascii')
                    base64_bytes = base64.b64encode(message_bytes)
                    f.write(base64_bytes)
                    f.write("\n".encode("ascii"))

    def import_list(self):
        file_path = filedialog.askopenfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, "rb") as f:
                lines = f.readlines()
            self.prompt_listbox.delete(0, tk.END)
            for line in lines:
                encoded = line.rstrip()
                decoded = base64.b64decode(encoded).decode('ascii')
                self.prompt_listbox.insert(tk.END, decoded)


root = tk.Tk()
app = TextExtractorApp(root)
root.mainloop()
