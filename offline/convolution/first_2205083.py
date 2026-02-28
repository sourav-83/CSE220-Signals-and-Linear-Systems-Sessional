import numpy as np
import matplotlib.pyplot as plt

class Signal:

    def __init__(self, INF):
        
        self.INF = INF
        self.signal = np.zeros(INF*2 + 1)

    def set_value_at_time(self, t, value):

        self.signal[self.INF + t] = value
    
    def shift(self, k):

        signal = Signal(self.INF)

        signal.signal = np.roll(self.signal, k)

        if k > 0:
            signal.signal[:k] = 0
        elif k < 0:
            signal.signal[k:] = 0
    
        return signal
    
    
    def add(self, other):

        signal = Signal(self.INF)
        signal.signal = np.add(self.signal, other.signal)
        return signal
    
    def multiply(self, scalar):

        signal = Signal(self.INF)
        signal.signal = np.multiply(self.signal, scalar)
        return signal
    
    def plot(self, title):
        
        plt.style.use("seaborn-v0_8")
        plt.figure()
        t = np.arange(-self.INF, self.INF+1)
        plt.stem(t, self.signal)
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("amplitude")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{title}.png')
        plt.show()


class LTI_System:

    def __init__(self, impulse_response):

        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal):

        coefficients = []
        impulses = []
        INF = input_signal.INF

        for k in range(-INF, INF+1):

            coefficient = input_signal.signal[k + INF]
            coefficients.append(coefficient)

            impulse = Signal(INF)
            impulse.set_value_at_time(k, 1)
            impulses.append(impulse)
        
        return impulses, coefficients
    
    def output(self, input_signal):

        INF = input_signal.INF
        x_coefficients = self.linear_combination_of_impulses(input_signal)[1]
        output_signal = Signal(INF)
        
        for k in range(-INF, INF+1):

            x_k = x_coefficients[INF + k]
            h_n_minus_k_signal = self.impulse_response.shift(k)
            output_signal = output_signal.add(h_n_minus_k_signal.multiply(x_k))

        return output_signal


if __name__ == "__main__":
    INF = 10

    # Input signal x(n)
    x = Signal(INF)
    x.set_value_at_time(-2, 1)
    x.set_value_at_time(0, 2)
    x.set_value_at_time(3, -1)

    x.plot("Input_Signal_x(n)")

    # Impulse response h(n)
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    h.plot("Impulse_Response_h(n)")

    # LTI System
    system = LTI_System(h)

    # Output
    y = system.output(x)
    y.plot("Output_Signal_y(n)")




      