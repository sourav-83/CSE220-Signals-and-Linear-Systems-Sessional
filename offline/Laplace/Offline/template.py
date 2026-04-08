import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

class SmartIrrigation:

    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):

        self.a = a
        self.b = b
        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0, t_max, dt)


    def u_step(self):

        return np.ones_like(self.t)
    
    def u_ramp(self):

        return 0.1 * self.t
    
    def u_sin(self):

        return np.sin(0.5 * self.t)
    
    def u_exponential(self):

        return 1.0 - np.exp(-0.3 * self.t)
    
    def u_pulse(self):
        
        return np.where(self.t < 5, 1.0, 0.0)

    def laplace_transform(self, f, s):
        
        integrand = f * np.exp(-s * self.t)
        integration_result = np.trapz(integrand, self.t)
        return integration_result
        

    def inverse_laplace(self, s_list, H_s_values):

        delta_omega = np.imag(s_list[1] - s_list[0])
        h_t = np.zeros_like(self.t, dtype=float)
        
        for j, t_val in enumerate(self.t):
            summation = np.sum(H_s_values * np.exp(s_list * t_val))
            
            h_t[j] = (delta_omega / (2 * np.pi)) * np.real(summation)
            
        return h_t

      
    def H_s(self, s, U_s):

        return (self.b / (s + self.a)) * U_s

    def steady_state(self, h):

        """Mean of last 5% of signal."""

        last_5_percent = max(1, int(.05 * len(h)))
        return np.mean(h[-last_5_percent:])
    
    def time_constant(self, h):

        """Time to first reach 63.2% of steady-state."""
        
        h_ss = self.steady_state(h)
        target = .632 * h_ss
        indices = np.where(h >= target)[0]
        if len(indices) > 0:
            return self.t[indices[0]]
        return None
    
    def rise_time(self, h):

        """Time to go from 10% to 90% of steady-state."""

        h_ss = self.steady_state(h)
        low_target = .1 * h_ss
        high_target = .9 * h_ss
        low_indices = np.where(h >= low_target)[0]
        high_indices = np.where(h >= high_target)[0]
        if len(low_indices) > 0  and len(high_indices) > 0:
            return self.t[high_indices[0]] - self.t[low_indices[0]]

    def settling_time(self, h):

        """Time after which h(t) stays permanently within ±2% of h_ss."""
        
        h_ss = self.steady_state(h)
        upper_bound = 1.02 * h_ss
        lower_bound = .98 * h_ss

        outside_indices = np.where((h > upper_bound) | (h < lower_bound))[0]
        if len(outside_indices) > 0:
            last_index = outside_indices[-1]
            if last_index + 1 < len(self.t):
                return self.t[last_index + 1]
            return None
        return self.t[0]

    def overshoot(self, h):

        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        
        h_ss = self.steady_state(h)
        h_max = np.max(h)
        if h_max > h_ss and h_ss != 0:
            return ((h_max - h_ss) / h_ss) * 100.0
        return 0.0

    def compute_metrics(self, h):
       
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    def euler_simulate(self, u):
        """
        Euler method for dh/dt = -a*h(t) + b*u(t)
        h[n+1] = h[n] + dt * (-a*h[n] + b*u[n])
        """
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            h[n + 1] = h[n] + self.dt * dhdt
        return h


#Change values of a, b to experiment with different system dynamics
system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

inputs = {
    "Step Input":        system.u_step(),
    "Ramp Input":        system.u_ramp(),
    "Sinusoidal Input":  system.u_sin(),
    "Exponential Input": system.u_exponential(),
    "Pulse Input":       system.u_pulse(),
}

# Bromwich contour parameters, set these values
c = .5 
W = 100.0
N = 4000
delta_omega = (2 * W) / N
k_values = np.arange(N)
omega_k = -W + k_values * delta_omega
s_list = c + 1j * omega_k

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")

    # --- Laplace --- set these values
    U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
    H_s_vals = np.array([system.H_s(s, U) for s, U in zip(s_list, U_s_vals)])
    h_laplace = system.inverse_laplace(s_list, H_s_vals)

    print(f"\n  ► {name}")
    metrics = system.compute_metrics(h_laplace)
    for k, v in metrics.items():
        print(f"      {k.replace('_',' ').title():<22}: {v}")

    # --- Euler ---
    h_euler = system.euler_simulate(u)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Smart Irrigation — {name}", fontsize=13, fontweight='bold')

    # Laplace subplot
    axes[0].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[0].plot(system.t, h_laplace, color=colors[idx], lw=2.2, label="Output h(t)")
    axes[0].set_title("Laplace Transform Simulation", fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Water Level / Input", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Euler subplot
    axes[1].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
    axes[1].set_title("Euler Method Simulation", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level / Input", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()