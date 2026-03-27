import numpy as np

def Factorize(n):
    factors = []
    f = 2
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 1
    if n > 1:
        factors.append(n)
    return factors

class DiscreteSignal:
    def __init__(self, data):         
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)
        
    def pad(self, new_length):
        length = len(self.data)
        if new_length <= length:
            padded = self.data[:new_length]
        else:
            padded = np.zeros(new_length, dtype=self.data.dtype)  
            padded[:length] = self.data
        return DiscreteSignal(padded)

    def interpolate(self, new_length):
        old_length = len(self.data)
        if old_length == new_length:
            return DiscreteSignal(self.data.copy())
        
        old_indices = np.linspace(0, old_length - 1, old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        real_interp = np.interp(new_indices, old_indices, self.data.real)
        imag_interp = np.interp(new_indices, old_indices, self.data.imag)
        return DiscreteSignal(real_interp + 1j * imag_interp)

class DFTAnalyzer:
    def compute_dft(self, signal: DiscreteSignal):
        N = len(signal)
        x = signal.data
        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
        return X

    def compute_idft(self, spectrum):
        N = len(spectrum) 
        x = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                x[n] += spectrum[k] * np.exp(1j * 2 * np.pi * k * n / N)
            x[n] = x[n] / N
        return x

class FastFourierTransform(DFTAnalyzer):
    def next_power_of_2(self, n):
        if n < 1: return 1
        power = 1
        while power < n:
            power <<= 1
        return power

    def is_power_of_2(self, n):
        return n > 0 and (n & (n - 1)) == 0

    def radix2_dit_fft(self, x):
        N = len(x)
        if N <= 1:
            return x.copy()
        even = self.radix2_dit_fft(x[0::2])  
        odd  = self.radix2_dit_fft(x[1::2])  
        k = np.arange(N // 2)
        twiddle = np.exp(-1j * 2 * np.pi * k / N) * odd
        X = np.empty(N, dtype=np.complex128)
        X[:N // 2] = even + twiddle
        X[N // 2:] = even - twiddle
        return X

    def radix2_dit_ifft(self, X):
        N = len(X)
        x = np.conj(self.radix2_dit_fft(np.conj(X))) / N
        return x

    def mixed_radix_fft(self, x):
        N = len(x)
        if N <= 1:
            return x.copy()
        factors = Factorize(N)
        if len(factors) == 1: 
            return super().compute_dft(DiscreteSignal(x))
        
        r = factors[0]
        m = N // r
        # Reshape and recursive call logic
        x_reshaped = x.reshape(m, r).T
        results = []
        for row in x_reshaped:
            results.append(self.mixed_radix_fft(row))
        x_res = np.array(results)

        n1 = np.arange(r).reshape(-1, 1)
        k2 = np.arange(m)
        twiddle = np.exp(-2j * np.pi * n1 * k2 / N)
        x_res = (x_res * twiddle).T
        
        final_results = []
        for row in x_res:
            final_results.append(self.mixed_radix_fft(row))
        
        return np.array(final_results).T.reshape(N)

    def compute_dft(self, signal: DiscreteSignal):
        
        x = signal.data
        N = len(x)

        if self.is_power_of_2(N):
            print("Using radix2 FFT")
            return self.radix2_dit_fft(x)
        else:
            print("Using mixed FFT")
            return self.mixed_radix_fft(x)
        
        N_original = len(signal)

        N_fft  = self.next_power_of_2(N_original)



        # Zero-pad to power of 2 if necessary

        # if N_fft != N_original:

        #     padded_signal = signal.pad(N_fft)

        #     x = padded_signal.data

        # else:

        #     x = signal.data.copy()



        # return self.radix2_dit_fft(x)



    def compute_idft(self, spectrum):

        N = len(spectrum)
        if self.is_power_of_2(N):
            return self.radix2_dit_ifft(spectrum)
        else:
            return np.conj(self.mixed_radix_fft(np.conj(spectrum))) / N
        



def circular_convolution(self, x: DiscreteSignal, h: DiscreteSignal):
    N = max(len(x), len(h))
    x_padded = x.pad(N)
    h_padded = h.pad(N)
    X = self.compute_dft(x_padded)
    H = self.compute_dft(h_padded)
    return self.compute_idft(X * H)

def cross_correlation(self, x: DiscreteSignal, h: DiscreteSignal):
    N = max(len(x), len(h))
    x_padded = x.pad(N)
    h_padded = h.pad(N)
    X = self.compute_dft(x_padded)
    H = self.compute_dft(h_padded)
    return self.compute_idft(np.conj(X) * H)

def linear_convolution(self, x: DiscreteSignal, h: DiscreteSignal):
    N = len(x) + len(h) - 1
    x_padded = x.pad(N)
    h_padded = h.pad(N)
    X = self.compute_dft(x_padded)
    H = self.compute_dft(h_padded)
    return self.compute_idft(X * H)