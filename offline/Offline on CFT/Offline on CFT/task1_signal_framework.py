import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')


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
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.values())
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
        """Generate a sine wave"""
        return amplitude * np.sin(2 * np.pi * frequency * self.t)

    def cosine(self, amplitude, frequency):
        """Generate a cosine wave"""
        return amplitude * np.cos(2 * np.pi * frequency * self.t)

    def square(self, amplitude, frequency):
        """Generate a square wave using sign of sine."""
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * self.t))

    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave"""
        ft = frequency * self.t
        return amplitude * 2 * (ft - np.floor(0.5 + ft))

    def triangle(self, amplitude, frequency):
        """Generate a triangle wave"""
        return (2 * amplitude / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * self.t))

    def cubic(self, coefficient):
        """Generate a cubic polynomial signal"""
        return coefficient * (self.t ** 3)

    def parabolic(self, coefficient):
        """Generate a parabolic signal"""
        return coefficient * (self.t ** 2)

    def rectangular(self, width):
        """Generate a rectangular window centered at t=0"""
        rect = np.where(np.abs(self.t) <= (width / 2.0), 1.0, 0.0)
        return rect

    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        p = np.where((self.t >= start) & (self.t <= end), 1.0, 0.0)
        return p


# =====================================================
# Composite Signal Class
# =====================================================

class CompositeSignal(ContinuousSignal):

    """Combines multiple signals into a single composite signal"""

    def __init__(self, t):
        super().__init__(t)
        self.components = []

    def add_component(self, signal):

        """Add a signal component to the composite signal"""

        self.components.append(signal)

    def values(self):
        
        """Sum all signal components"""

        return np.sum(self.components, axis=0)


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
        """Compute real and imaginary parts of the CFT"""
        
        x_vals = self.signal.values()
        X_real = []
        X_imaginary = []

        """
        X(f) = integral(x(t) * e^(-j2pi*f*t) dt)
             = integral(x(t) * [cos(2pi*f*t) - j*sin(2pi*f*t)] dt)
        """

        for f in self.frequencies:
            
            integrand_real = x_vals * np.cos(2 * np.pi * f * self.t)
            val_real = np.trapz(integrand_real, self.t)
            X_real.append(val_real)

            
            integrand_imaginary = x_vals * -np.sin(2 * np.pi * f * self.t)
            val_imaginary = np.trapz(integrand_imaginary, self.t)
            X_imaginary.append(val_imaginary)

        return np.array(X_real), np.array(X_imaginary)

    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        
        """
        X_real, X_imaginary = self.compute_cft()
        magnitude = np.sqrt(X_real**2 + X_imaginary**2) 

        plt.figure(figsize=(10, 4))
        plt.plot(self.frequencies, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("magnitude |X(f)|")
        plt.title("CFT magnitude Spectrum")
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
        X_real, X_imag = self.spectrum
        x_reconstructed = []

        """
        x(t) = integral(X(f) * e^(j2pi*f*t) df)
        Integrand Real Part = X_real * cos(2pi*f*t) - X_imag * sin(2pi*f*t) 

        """

        for t_val in self.t:
        
            x_real_multiply_cos = X_real * np.cos(2 * np.pi * self.frequencies * t_val)
            
            
            x_imaginary_multiply_sin = X_imag * np.sin(2 * np.pi * self.frequencies * t_val)
            
            
            integrand = x_real_multiply_cos - x_imaginary_multiply_sin
            
            val = np.trapz(integrand, self.frequencies)

            x_reconstructed.append(val)

        return np.array(x_reconstructed)


# =====================================================
# Main Execution (Task 1)
# =====================================================

def main():

    # Define time axis
    t = np.linspace(-4, 4, 3000)
    gen = SignalGenerator(t)

    composite = CompositeSignal(t)
    composite.add_component(gen.sine(2, 1))
    composite.add_component(gen.cosine(0.5, 3))
    composite.add_component(gen.square(1, 1))
    composite.add_component(gen.cubic(1) * gen.rectangular(2))


    composite.plot("Composite Signal")

    frequencies = np.linspace(-10, 10, 500) 


    cft = CFTAnalyzer(composite, t, frequencies)
    cft.plot_spectrum()


    spectrum_data = cft.compute_cft() 
    icft = InverseCFT(spectrum_data, frequencies, t)
    x_rec = icft.reconstruct()

    plt.plot(t, composite.values(), label="Original")
    plt.plot(t, x_rec, '--', label="Reconstructed")
    plt.legend()
    plt.title("Reconstruction using ICFT")
    plt.show()

if __name__ == "__main__":
    main()