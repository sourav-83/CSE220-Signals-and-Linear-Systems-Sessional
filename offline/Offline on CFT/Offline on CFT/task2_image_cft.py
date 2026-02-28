import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

plt.style.use('seaborn-v0_8')

# =====================================================
# Continuous Image Class
# =====================================================

class ContinuousImage:
    """
    Represents an image as a continuous 2D signal f(x, y).
    """

    def __init__(self, image_path):
        self.image = imageio.imread(image_path, mode='L')
        self.image = self.image / np.max(self.image)

        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.grid(True)
        plt.savefig(f'{title}.png')
        plt.show()


# =====================================================
# 2D Continuous Fourier Transform Class
# =====================================================

class CFT2D:
    """
    Computes 2D Continuous Fourier Transform using numerical integration.
    """

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        """
        Calculates the amount of each frequency (u, v) present in the image.
        """
        # Initialize containers for the complex coefficients F(u, v)
        real = np.zeros_like(self.I, dtype=np.float64)
        imaginary = np.zeros_like(self.I, dtype=np.float64)
        
        # Create a coordinate grid representing the spatial plane (x, y)
        X_axis, Y_axis = np.meshgrid(self.x, self.y)
        
        # Iteratint through every frequency coordinate (u, v)
        for i in range(self.I.shape[0]):
            for j in range(self.I.shape[1]):
                u = self.x[j] # Horizontal frequency
                v = self.y[i] # Vertical frequency
                
                # Kernel: e^(-j * 2π * (ux + vy))
                kernel = np.exp(-2j * np.pi * (u * X_axis + v * Y_axis))
                
                # Multiply signal f(x, y) by the complex wave
                total_term = self.I * kernel
                
                # Perform the Double Integral: integration( integration( f(x, y) * kernel dx dy))
                real[i, j] = np.trapz(np.trapz(total_term.real, self.x, axis=1), self.y, axis=0)

                imaginary[i, j] = np.trapz(np.trapz(total_term.imag, self.x, axis=1), self.y, axis=0)

        return real, imaginary

    def plot_mgntde(self):

        # magnitude = sqrt(Real² + Imaginary²)

        real, imaginary = self.compute_cft()
        mgntde = np.sqrt(real**2 + imaginary**2)
        
        # compared to higher frequencies. log(1+m) compresses the dynamic range.
        plt.imshow(np.log(mgntde + 1), cmap='inferno')
        plt.title("Log magnitude Spectrum")
        plt.axis('off')
        plt.legend()
        plt.show()
        plt.grid(True)
        plt.savefig("Log_magnitude_Spectrum.png")


# =====================================================
# Frequency Filtering (Low-Pass)
# =====================================================

class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        """
        Filters out noise. Noise is usually high-frequency (fast changes).
        """
        rows, cols = real.shape
        cx, cy = rows//2, cols//2 

        for i in range(rows):
            for j in range(cols):
                # distance > cutoff means it's a high frequency
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real[i,j] = 0
                    imag[i,j] = 0
        return real, imag



# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================

class InverseCFT2D:
    """
    Reconstructs f(x, y) by summing all complex waves scaled by F(u, v).
    """

    def __init__(self, real, imag, x, y):
        self.real = real
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        # Initialize complex spatial image
        reconstructed_fxy = np.zeros((len(self.y), len(self.x)), dtype=complex)
        
        # Frequency grid (U, V)
        U, V = np.meshgrid(self.x, self.y)
        spectrum = self.real + 1j * self.imag
        
        # Summing up waves for every pixel (x, y)
        for i in range(len(self.y)):
            for j in range(len(self.x)):
                x = self.x[j]
                y = self.y[i]
                
                # Inverse Kernel: e^(+j * 2π * (ux + vy))
                inverse_kernel = np.exp(2j * np.pi * (U * x + V * y))
                
                # Integrate across the frequency domain (u, v)
                total_term = spectrum * inverse_kernel
                reconstructed_fxy[i, j] = np.trapz( np.trapz(total_term, self.x, axis=1), self.y, axis=0)

        return np.real(reconstructed_fxy)



# =====================================================
# Main Execution
# =====================================================

def main():
    img = ContinuousImage("noisy_image.png")
    img.show("Original Image")

    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    cft2d.plot_mgntde()


    filt = FrequencyFilter()
    real_f, imag_f = filt.low_pass(real, imag, cutoff=40)


    icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
    denoised = icft2d.reconstruct()

    plt.imshow(denoised, cmap='gray')
    plt.title("Reconstructed (Denoised) Image")
    plt.axis('off')
    plt.savefig("Reconstructed.png")
    plt.show()

if __name__ == "__main__":
    main()