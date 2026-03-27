import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, DiscreteSignal, FastFourierTransform


class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")
        
        self.samplerate = 0
        self.original_audio = None
        self.processed_audio = None
        
        # UI Layout 
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
        
        tk.Button(top_frame, text="Load WAV File", command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play", command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio", command=sd.stop).pack(side=tk.LEFT, padx=10)
        
        # Toggle Switch
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text="Algorithm: ").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="DFT (Slow)", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT (Fast)", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # Equalizer Sliders
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=20, padx=20)
        
        self.sliders = []
        labels = ["Low", "Low-Mid", "Mid", "High-Mid", "High"]
        for i in range(5):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                self.samplerate, data = wav.read(file_path)
                
                # Normalize to float [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                self.original_audio = data
                self.processed_audio = None
                duration = len(data) / self.samplerate
                print(f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return
        
        print("Starting processing...")

        # Get slider gain values for the 5 bands
        gains = [s.get() for s in self.sliders]

        # Choose analyzer based on toggle
        if self.use_fft.get():
            analyzer = FastFourierTransform()
        else:
            analyzer = DFTAnalyzer()

        # Block processing parameters
        chunk_size = 1024
        audio = self.original_audio
        num_samples = len(audio)
        output_audio = np.zeros(num_samples, dtype=np.float32)

        num_chunks = int(np.ceil(num_samples / chunk_size))

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, num_samples)
            chunk = audio[start:end]

            # Zero-pad the last chunk if it is shorter than chunk_size
            original_chunk_len = len(chunk)
            if original_chunk_len < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - original_chunk_len))

            # Frequency Analysis 
            signal = DiscreteSignal(chunk)
            spectrum = analyzer.compute_dft(signal) 

            N = len(spectrum)
            half = N // 2 


            band_size = half // 5

            for b in range(5):
                bin_start = b * band_size
                # Last band gets any remaining bins up to the Nyquist bin
                bin_end = (b + 1) * band_size if b < 4 else half

                gain = gains[b]

                # Apply gain to positive frequencies (including DC for band 0)
                spectrum[bin_start:bin_end] *= gain

                neg_start = N - bin_end
                neg_end = N - bin_start
                if neg_start < neg_end and neg_start > 0 and neg_end <= N:
                    spectrum[neg_start:neg_end] *= gain

            # Reconstruction 
            reconstructed = analyzer.compute_idft(spectrum)

            # Take real part and trim back to original chunk length
            chunk_out = np.real(reconstructed[:original_chunk_len]).astype(np.float32)
            output_audio[start:end] = chunk_out

            if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
                print(f"  Processed chunk {i+1}/{num_chunks}")

        # Clip to [-1, 1] to prevent distortion/overflow
        output_audio = np.clip(output_audio, -1.0, 1.0)

        self.processed_audio = output_audio

        # Playback
        sd.stop()
        sd.play(self.processed_audio, self.samplerate)
        print("Playback started.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()