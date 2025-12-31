import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

class StochasticProcess:
    def __init__(self, T, N, n):
        self.process_type = None
        self.T = T
        self.N = N
        self.dt = T / N
        self.n = n
        self.data = None
        self.process_params = {}
        self.process_map = {
            "GBM": self.geometric_brownian_motion,
            "MJD": self.merton_jump_diffusion,
        }

    def geometric_brownian_motion(self, St, mu, sigma):
        prices = [St]
        dWt = np.random.randn(self.N) * np.sqrt(self.dt)
        for i in range(self.N):
            St += mu * St * self.dt + sigma * St * dWt[i]
            prices.append(St)
        return prices
    
    def merton_jump_diffusion(self, St, mu, sigma, lam, muJ, sigmaJ):
        prices = [St]
        dWt = np.random.randn(self.N) * np.sqrt(self.dt)
        dNt = np.random.poisson(lam * self.dt, size=self.N)
        for i in range(self.N):
            J = np.random.normal(muJ, sigmaJ) if dNt[i] > 0 else 0
            St += mu * St * self.dt + sigma * St * dWt[i] + St * (np.exp(J) - 1) * dNt[i]
            prices.append(St)
        return prices
    
    def define_process(self, process_type, **params):
        """Set the process type and store default parameters internally.

        Example:
            obj.define_process("GBM", St=100, mu=0.05, sigma=0.2)

        Stored parameters are used by `simulate` unless overridden by kwargs
        passed to `simulate`.
        """
        if process_type not in self.process_map:
            raise ValueError(f"Unknown process type: {process_type}")
        self.process_type = process_type
        # store a shallow copy of params
        self.process_params = dict(params)

    def simulate(self, **kwargs):
        if self.process_type not in self.process_map:
            raise ValueError(f"Unknown process type: {self.process_type}")
        
        process_func = self.process_map[self.process_type]
        # Merge stored parameters with provided kwargs; explicit kwargs override
        params = dict(self.process_params)
        params.update(kwargs)

        simulations = [process_func(**params) for _ in range(self.n)]
        return simulations
    
    def visualize_process(self, simulations):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        for path in simulations:
            sns.lineplot(x=range(len(path)), y=path, alpha=0.6)
        plt.title(f"{self.process_type} Simulation ({self.n} paths)")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.show()

    def visualize_mc_analysis(self, simulations):
        final_values = [path[-1] for path in simulations]
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.histplot(final_values, bins=30, kde=True, alpha=0.7)
        plt.title(f"Monte Carlo Analysis of {self.process_type} Final Values")
        plt.xlabel("Final Value")
        plt.ylabel("Frequency")
        plt.show()

    def mc_analysis(self, simulations):
        final_values = [path[-1] for path in simulations]
        mean = np.mean(final_values)
        std_dev = np.std(final_values)
        conf_int = norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(self.n))
        return mean, std_dev, conf_int
    
    def import_data(self, filepath, header = True, parse_dates = True):
        self.data = pd.read_csv(filepath, header=header, parse_dates=parse_dates)

    

# Example usage:
if __name__ == "__main__":
    sp = StochasticProcess(T=1, N=100, n=1000)
    sp.define_process("MJD", St=100, mu=0.05, sigma=0.2, lam=0.1, muJ=-0.02, sigmaJ=0.1)
    sims = sp.simulate()
    sp.visualize_process(sims)
    sp.visualize_mc_analysis(sims)
    mean, std_dev, conf_int = sp.mc_analysis(sims)
    print(f"Mean final value: {mean}, Std Dev: {std_dev}, 95% CI : {conf_int}")
