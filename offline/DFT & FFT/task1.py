import tkinter as tk
import numpy as np
import math
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierTransform

class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")
        
        # UI Layout
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        # Buttons
        tk.Button(control_frame, text="Clear Canvas", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles", command=self.run_transform).pack(side=tk.LEFT, padx=5)
        
        # Toggle Switch (Radio Buttons)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points = []
        self.drawing = False
        self.fourier_coeffs = None
        self.is_animating = False
        self.after_id = None

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    def draw_epicycle(self, x, y, radius):
        """
        Helper method for students to draw a circle (epicycle).
        x, y: Center coordinates
        radius: Radius of the circle
        """
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline="blue", tags="epicycle")

    def run_transform(self):

        if not self.points:
            return

        # CENTER DRAWING 
        mean_x = np.mean([x for x, y in self.points])
        mean_y = np.mean([y for x, y in self.points])

        complex_samples = [
            complex(x - mean_x, y - mean_y)
            for x, y in self.points
        ]

        signal = DiscreteSignal(complex_samples)

        if self.use_fft.get():
            analyzer = FastFourierTransform()
            
            ###

            # N = len(signal)
            # N_fft = analyzer.next_power_of_2(N)
            # signal = signal.interpolate(N_fft)

            ###

            coeffs = analyzer.compute_dft(signal)
        else:
            analyzer = DFTAnalyzer()
            coeffs = analyzer.compute_dft(signal)

        N = len(coeffs)

        self.fourier_coeffs = [(k, coeffs[k]) for k in range(N)]
        self.fourier_coeffs.sort(key=lambda x: abs(x[1]), reverse=True)

        self.center_x = mean_x
        self.center_y = mean_y

        self.time_step = 0
        self.trace = []
        self.is_animating = True

        self.update_frame()

        

    def animate_epicycles(self, center_offset):
        self.is_animating = True
        self.time_step = 0
        self.num_frames = self.N
        
        self.center_offset = center_offset
        self.update_frame()

    def update_frame(self):

        if not self.is_animating or self.fourier_coeffs is None:
            return

        self.canvas.delete("epicycle")

        N = len(self.fourier_coeffs)
        t = self.time_step

        current_x, current_y = 0, 0

        for k, Xk in self.fourier_coeffs:

            prev_x, prev_y = current_x, current_y

            radius = abs(Xk) / N
            phase = np.angle(Xk)

            angle = 2 * np.pi * k * t / N + phase

            current_x += radius * np.cos(angle)
            current_y += radius * np.sin(angle)

            if radius > 0.5:
                self.draw_epicycle(
                    prev_x + self.center_x,
                    prev_y + self.center_y,
                    radius
                )

                self.canvas.create_line(
                    prev_x + self.center_x,
                    prev_y + self.center_y,
                    current_x + self.center_x,
                    current_y + self.center_y,
                    fill="red",
                    tags="epicycle"
                )

        cx = current_x + self.center_x
        cy = current_y + self.center_y

        self.trace.append((cx, cy))

        if len(self.trace) > 1:
            self.canvas.create_line(
                self.trace[-2][0], self.trace[-2][1],
                self.trace[-1][0], self.trace[-1][1],
                fill="black", width=2, tags="epicycle"
            )

        r = 3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                fill="red", outline="red",
                                tags="epicycle")

        self.time_step = (self.time_step + 1) % N
        self.after_id = self.root.after(20, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()