import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
            "SBM": self.simple_brownian_motion,
            "GBM": self.geometric_brownian_motion,
            "MJD": self.merton_jump_diffusion,
        }

    def simple_brownian_motion(self):
        values = [0.0]
        dWt = np.random.randn(self.N) * np.sqrt(self.dt)
        Wt = 0.0
        for i in range(self.N):
            Wt += dWt[i]
            values.append(Wt)
        return values

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
        # Find mode by evaluating KDE at grid points and finding maximum
        kde = stats.gaussian_kde(final_values)
        x_grid = np.linspace(min(final_values), max(final_values), 1000)
        kde_values = kde(x_grid)
        mode = x_grid[np.argmax(kde_values)]
        std_dev = np.std(final_values, ddof=1)
        lower, upper = np.percentile(final_values, [2.5, 97.5])
        return mean, mode, std_dev, (lower, upper)

    def import_data(self, filepath, header=True, parse_dates=True):
        self.data = pd.read_csv(filepath, header=header, parse_dates=parse_dates)

    def set_data(self, data_df):
        """Set data directly from a pandas DataFrame."""
        self.data = data_df

    def derive_bayesian_posterior(self):
        """Derive Bayesian posterior process using imported data.
        Currently supports GBM. Estimates mu and sigma from data 'Close' column.
        Returns a new StochasticProcess with posterior parameters, with N matching data intervals.
        """
        if self.data is None:
            raise ValueError("No data imported. Use import_data() first.")
        
        if self.process_type != "GBM":
            raise NotImplementedError("Bayesian posterior currently only implemented for GBM.")
        
        # Extract closing prices (case-insensitive)
        if 'Close' in self.data.columns:
            prices = self.data['Close'].values
        elif 'close' in self.data.columns:
            prices = self.data['close'].values
        else:
            raise ValueError("Data must contain 'Close' or 'close' column.")
        
        # Number of intervals = number of data points - 1
        new_N = len(prices) - 1
        if new_N <= 0:
            raise ValueError("Data must contain at least two points to estimate process.")
        
        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        n_obs = len(log_returns)

        # Estimate parameters from data - scale to match time horizon
        mu_data = np.mean(log_returns) * (new_N / self.T)
        sigma_data = np.std(log_returns, ddof=1) * np.sqrt(new_N / self.T)

        # Get prior parameters
        mu_prior = self.process_params.get('mu', 0.05)
        sigma_prior = self.process_params.get('sigma', 0.2)

        # Precision-weighted Bayesian update for mu
        # Prior precision (assume prior variance = sigma_prior^2)
        prior_precision_mu = 1.0 / (sigma_prior ** 2)
        # Data precision (variance of sample mean = sigma_data^2 / n_obs)
        data_precision_mu = n_obs / (sigma_data ** 2)
        # Posterior mu weighted by precision
        mu_posterior = (prior_precision_mu * mu_prior + data_precision_mu * mu_data) / (prior_precision_mu + data_precision_mu)

        # Precision-weighted Bayesian update for sigma
        # Use inverse variance weighting with assumed prior degrees of freedom
        prior_dof = 10  # Assumed prior sample size
        data_dof = n_obs
        # Weight by effective sample sizes (degrees of freedom)
        sigma_posterior = (prior_dof * sigma_prior + data_dof * sigma_data) / (prior_dof + data_dof)
        
        # Create new StochasticProcess for posterior with N matching data intervals
        posterior = StochasticProcess(T=self.T, N=new_N, n=self.n)
        posterior.define_process("GBM", St=self.process_params['St'], 
                                 mu=mu_posterior, sigma=sigma_posterior)
        posterior.data = self.data  # Keep reference to data
        return posterior