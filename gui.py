import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

from train import train_model

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SLM Trainer")

        self.file_path = tk.StringVar()

        tk.Button(root, text="Select Corpus", command=self.select_file).pack(pady=5)
        tk.Label(root, textvariable=self.file_path).pack()
        tk.Button(root, text="Start Training", command=self.start_training).pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.model_info = tk.Text(root, height=5, width=40)
        self.model_info.pack(pady=5)

    def select_file(self):
        path = filedialog.askopenfilename()
        if path:
            self.file_path.set(path)

    def start_training(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "No corpus selected")
            return
        threading.Thread(target=self.train).start()

    def train(self):
        with open(self.file_path.get(), "r") as f:
            text = f.read()
        model, vocab, losses = train_model(text, epochs=5)
        self.ax.clear()
        self.ax.plot(range(1, len(losses) + 1), losses, marker="o")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.canvas.draw()

        num_params = sum(p.numel() for p in model.parameters())
        info = f"Model: {model.__class__.__name__}\nParameters: {num_params}\nVocab size: {len(vocab)}"
        self.model_info.delete("1.0", tk.END)
        self.model_info.insert(tk.END, info)

        ckpt_path = "checkpoints/model.pth"
        torch.save({"model_state_dict": model.state_dict(), "vocab": vocab}, ckpt_path)
        messagebox.showinfo("Training complete", f"Model saved to {ckpt_path}")


def main():
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
