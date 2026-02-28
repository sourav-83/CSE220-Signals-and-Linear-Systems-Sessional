import numpy as np
import matplotlib.pyplot as plt
 
# =====================================================
# Abstract Base Class for Continuous-Time Signals
# =====================================================
class ContinuousSignal:
    """
    Abstract base class for all continuous-time signals.
    Every signal must be defined over a time axis t.
    """
 
    def __init__(self, t):
        self.t = t
 
    def values(self):
        """
        Returns the signal values evaluated over time axis t.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
 
    def plot(self, title="Signal"):
        """
        Plot the signal in the time domain.
        """
        y = np.asarray(self.values())
        if np.iscomplexobj(y):
            plt.plot(self.t, y.real, label="Real")
            plt.plot(self.t, y.imag, "--", label="Imag")
            plt.legend()
        else:
            plt.plot(self.t, y)
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()
 
 
# =====================================================
# Signal Generator Class
# =====================================================
class SignalGenerator(ContinuousSignal):
    """
    Generates various continuous-time signals.
    Each method returns a numpy array of signal samples.
    """
 
    def sine(self, amplitude, frequency):
        """Generate a sine wave."""
        y = amplitude * np.sin(2 * np.pi * frequency * self.t)
        return y
 
    def cosine(self, amplitude, frequency):
        """Generate a cosine wave."""
        y = amplitude * np.cos(2 * np.pi * frequency * self.t)
        return y
 
    def square(self, amplitude, frequency):
        """Generate a square wave using sign of sine."""
        y = amplitude * np.sign(np.sin(2 * np.pi * frequency * self.t))
        return y
 
    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave."""
        y = 2 * amplitude * (frequency * self.t - np.floor(0.5 + frequency * self.t))
        return y
 
    def triangle(self, amplitude, frequency):
        """Generate a triangle wave."""
        y = amplitude * (2 * np.abs(self.sawtooth(1, frequency)) - 1)
        return y
 
    def cubic(self, coefficient):
        """Generate a cubic polynomial signal."""
        y = coefficient * np.power(self.t, 3)
        return y
 
    def parabolic(self, coefficient):
        """Generate a parabolic signal."""
        y = coefficient * np.power(self.t, 2)
        return y
 
    def rectangular(self, width):
        """Generate a rectangular window centered at t=0."""
        y = np.where(np.abs(self.t) <= width / 2,
            1,
            0
        )
        return y
 
    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        y = np.where(np.logical_and(start <= self.t, self.t <= end),
            1,
            0
        )
        return y
 
 
# =====================================================
# Composite Signal Class
# =====================================================
class CompositeSignal(ContinuousSignal):
    """
    Combines multiple signals into a single composite signal.
    """
 
    def __init__(self, t):
        super().__init__(t)
        self.components = []
 
    def add_component(self, signal):
        """
        Add a signal component to the composite signal.
        """
        self.components.append(signal)
 
    def values(self):
        """
        Sum all signal components.
        """
        result = np.zeros(len(self.t), dtype=np.complex128)
        for component in self.components:
            component_vals = component.values() if isinstance(component, ContinuousSignal) else component
            component_vals = np.asarray(component_vals)
            if component_vals.shape != self.t.shape:
                raise ValueError("All components must match the time axis shape.")
            result = result + component_vals
        return result
 
 
# =====================================================
# Continuous Fourier Transform Analyzer
# =====================================================
class CFTAnalyzer:
    """
    Computes the Continuous Fourier Transform (CFT)
    using numerical integration (np.trapz).
    """
 
    def __init__(self, signal, t, frequencies):
        self.signal = signal
        self.t = t
        self.frequencies = frequencies
        
 
    def compute_cft(self):
        """
        Compute the (possibly complex) Continuous Fourier Transform.
 
        Uses: X(f) = \int x(t) e^{-j 2\pi f t} dt
        """
        spectrum = np.empty(len(self.frequencies), dtype=np.complex128)
        sig_vals = np.asarray(self.signal.values(), dtype=np.complex128)
 
        for i, f in enumerate(self.frequencies):
            kernel = np.exp(-1j * 2 * np.pi * f * self.t)
            spectrum[i] = np.trapezoid(sig_vals * kernel, self.t)
 
        return spectrum
 
    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        """
        spectrum = self.compute_cft()
        magnitude = np.abs(spectrum)
        argument = np.angle(spectrum)
        plt.plot(self.frequencies, magnitude, label="magnitude")
        plt.plot(self.frequencies, argument, "--", label="argument")
        plt.xlabel("frequency")
        plt.title("Magnitude and argument graph of fourier transform")
        plt.grid(True)
        plt.show()
 
 
 
# =====================================================
# Inverse Continuous Fourier Transform
# =====================================================
class InverseCFT:
    """
    Reconstructs time-domain signal using ICFT.
    """
 
    def __init__(self, spectrum, frequencies, t):
        self.spectrum = spectrum
        self.frequencies = frequencies
        self.t = t
 
    def reconstruct(self):
        """
        Perform inverse CFT using numerical integration.
        """
        spectrum = np.asarray(self.spectrum, dtype=np.complex128)
        signal = np.empty(len(self.t), dtype=np.complex128)
 
        for i, tt in enumerate(self.t):
            kernel = np.exp(1j * 2 * np.pi * self.frequencies * tt)
            signal[i] = np.trapezoid(spectrum * kernel, self.frequencies)
 
        return signal
 
 
 
# =====================================================
# Main Execution (Task 1)
# =====================================================
t = np.linspace(-4, 4, 3000)
gen = SignalGenerator(t)
 
composite = CompositeSignal(t)
composite.add_component(gen.sine(2, 1))
composite.add_component(gen.cosine(0.5, 3))
composite.add_component(gen.square(1 + 2j, 1))
composite.add_component(gen.cubic(2 - 1j) * gen.rectangular(2))
 
composite.plot("Composite Signal")
 
frequencies = np.linspace(-10, 10, 3000)
cft = CFTAnalyzer(composite, t, frequencies)
cft.plot_spectrum()
 
icft = InverseCFT(cft.compute_cft(), frequencies, t)
x_rec = icft.reconstruct()
 
x_orig = np.asarray(composite.values())
plt.plot(t, x_orig.real, label="Original (real)")
if np.iscomplexobj(x_orig):
    plt.plot(t, x_orig.imag, "--", label="Original (imag)")
 
plt.plot(t, x_rec.real, label="Reconstructed (real)")
if np.iscomplexobj(x_rec):
    plt.plot(t, x_rec.imag, "--", label="Reconstructed (imag)")
plt.legend()
plt.title("Reconstruction using ICFT")
plt.show()