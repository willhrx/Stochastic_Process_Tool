import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stochastic_class import StochasticProcess

# Set seaborn style globally
sns.set_style("whitegrid")

# Page configuration
st.set_page_config(page_title="Stochastic Process Simulator", layout="wide")
st.title("Stochastic Process Simulator")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Process selection
process_type = st.sidebar.selectbox(
    "Select Process Type",
    ["SBM", "GBM", "MJD"],
    help="SBM: Simple Brownian Motion (Wiener Process)\nGBM: Geometric Brownian Motion\nMJD: Merton Jump Diffusion"
)

# Time and simulation parameters
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    T = st.number_input("Time Horizon (T)", value=1.0, step=0.1, min_value=0.1)
with col2:
    N = st.number_input("Time Steps (N)", value=100, step=10, min_value=10)
with col3:
    n_sims = st.number_input("Number of Paths", value=100, step=10, min_value=10)

st.sidebar.divider()

# Process-specific parameters
st.sidebar.header("Process Parameters")

if process_type == "SBM":
    st.sidebar.info("Simple Brownian Motion has no parameters - it's a pure Wiener process starting at 0.")
else:
    # Initial stock price
    St = st.sidebar.number_input("Initial Stock Price (S₀)", value=100.0, step=1.0, min_value=0.1)

    # Drift and volatility
    mu = st.sidebar.number_input("Drift (μ)", value=0.05, step=0.01, format="%.4f")
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01, format="%.4f")

    # Additional parameters for MJD
    if process_type == "MJD":
        st.sidebar.subheader("Jump Parameters")
        lam = st.sidebar.number_input("Jump Intensity (λ)", value=0.1, step=0.01, min_value=0.0, format="%.4f")
        muJ = st.sidebar.number_input("Jump Mean (μⱼ)", value=-0.02, step=0.01, format="%.4f")
        sigmaJ = st.sidebar.number_input("Jump Std Dev (σⱼ)", value=0.1, step=0.01, min_value=0.0, format="%.4f")

st.sidebar.divider()

# Data import section
st.sidebar.header("Import Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload OHLCV CSV file",
    type=["csv"],
    help="CSV file with OHLCV data"
)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Monte Carlo Analysis", "Data Import", "Bayesian Posterior"])

with tab1:
    st.header("Stochastic Process Simulation")
    
    if st.button("Run Simulation", key="run_sim"):
        with st.spinner("Running simulation..."):
            # Create and configure the process
            sp = StochasticProcess(T=T, N=N, n=n_sims)

            if process_type == "SBM":
                sp.define_process("SBM")
            elif process_type == "GBM":
                sp.define_process("GBM", St=St, mu=mu, sigma=sigma)
            else:  # MJD
                sp.define_process("MJD", St=St, mu=mu, sigma=sigma, lam=lam, muJ=muJ, sigmaJ=sigmaJ)

            # Run simulations
            simulations = sp.simulate()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for path in simulations:
                ax.plot(range(len(path)), path, alpha=0.6, linewidth=0.8)
            
            ax.set_title(f"{process_type} Simulation ({n_sims} paths)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Time step", fontsize=11)
            ax.set_ylabel("Value", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display statistics
            final_values = np.array([path[-1] for path in simulations])
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Final Value", f"${final_values.mean():.2f}")
            with col2:
                st.metric("Std Deviation", f"${final_values.std():.2f}")
            with col3:
                st.metric("Min/Max", f"${final_values.min():.2f} / ${final_values.max():.2f}")

with tab2:
    st.header("Monte Carlo Analysis")
    
    if st.button("Run Analysis", key="run_mc"):
        with st.spinner("Running Monte Carlo analysis..."):
            # Create and configure the process
            sp = StochasticProcess(T=T, N=N, n=n_sims)

            if process_type == "SBM":
                sp.define_process("SBM")
            elif process_type == "GBM":
                sp.define_process("GBM", St=St, mu=mu, sigma=sigma)
            else:  # MJD
                sp.define_process("MJD", St=St, mu=mu, sigma=sigma, lam=lam, muJ=muJ, sigmaJ=sigmaJ)

            # Run simulations
            simulations = sp.simulate()
            final_values = [path[-1] for path in simulations]
            
            # Monte Carlo analysis
            mean, mode, std_dev, conf_int = sp.mc_analysis(simulations)
            
            # Distribution plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(final_values, bins=30, kde=True, alpha=0.7, ax=ax)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean:.2f}')
            ax.axvline(mode, color='blue', linestyle='--', linewidth=2, label=f'Gaussian KDE: ${mode:.2f}')
            ax.axvline(conf_int[0], color='orange', linestyle='--', linewidth=2, label=f'95% CI: ${conf_int[0]:.2f} - ${conf_int[1]:.2f}')
            ax.axvline(conf_int[1], color='orange', linestyle='--', linewidth=2)
            
            ax.set_title(f"Monte Carlo Analysis of {process_type} Final Values", fontsize=14, fontweight="bold")
            ax.set_xlabel("Final Value", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"${mean:.2f}")
            with col2:
                st.metric("Gaussian KDE", f"${mode:.2f}")
            with col3:
                st.metric("Standard Deviation", f"${std_dev:.2f}")
            with col4:
                st.metric("95% CI", f"${conf_int[0]:.2f} to ${conf_int[1]:.2f}")
            
            # Additional statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", f"${np.min(final_values):.2f}")
            with col2:
                st.metric("Max", f"${np.max(final_values):.2f}")
            with col3:
                st.metric("Median", f"${np.median(final_values):.2f}")
            with col4:
                st.metric("Skewness", f"{pd.Series(final_values).skew():.4f}")

with tab3:
    st.header("OHLCV Data Import & Analysis")
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df  # Store data in session state
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            with col2:
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe())
            
            # Plot closing prices if available
            if 'Close' in df.columns or 'close' in df.columns:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df[close_col], linewidth=2, color='steelblue')
                ax.set_title("Historical Closing Prices", fontsize=14, fontweight="bold")
                ax.set_xlabel("Time", fontsize=11)
                ax.set_ylabel("Price", fontsize=11)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        st.info("Upload a CSV file with OHLCV data to view and analyze it.")
        st.write("**Expected columns:** Date, Open, High, Low, Close, Volume")

with tab4:
    st.header("Bayesian Posterior Process")

    # Button to derive posterior process
    if process_type == "GBM":
        if 'uploaded_data' in st.session_state:
            if st.button("Derive Bayesian Posterior Process"):
                with st.spinner("Deriving posterior process..."):
                    df = st.session_state['uploaded_data']
                    # Determine number of intervals from data
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices = df[close_col].values
                    new_N = len(prices) - 1
                    # Create prior process with N matching data intervals
                    sp_prior = StochasticProcess(T=T, N=new_N, n=20)
                    sp_prior.define_process("GBM", St=St, mu=mu, sigma=sigma)
                    sp_prior.set_data(df)
                    try:
                        sp_posterior = sp_prior.derive_bayesian_posterior()
                        st.session_state['sp_prior'] = sp_prior
                        st.session_state['sp_posterior'] = sp_posterior
                        st.session_state['uploaded_data_for_posterior'] = df
                        st.success("Bayesian posterior process derived successfully!")
                    except Exception as e:
                        st.error(f"Error deriving posterior: {e}")
        else:
            st.info("Please upload data in the 'Data Import' tab first.")
    else:
        st.info("Bayesian posterior derivation is currently only supported for GBM process type. Change process type in the sidebar.")

    st.divider()

    if 'sp_prior' in st.session_state and 'sp_posterior' in st.session_state:
        sp_prior = st.session_state['sp_prior']
        sp_posterior = st.session_state['sp_posterior']
        data = st.session_state['uploaded_data']
        
        st.subheader("Process Definitions")
        
        # Display equations and parameters
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Prior Process")
            st.latex(r'\mathrm{d}S_t = \mu S_t \mathrm{d}t + \sigma S_t \mathrm{d}W_t')
            st.write(f"Parameters: $\\mu = {sp_prior.process_params['mu']:.4f}$, $\\sigma = {sp_prior.process_params['sigma']:.4f}$")
            
        with col2:
            st.markdown("### Posterior Process")
            st.latex(r'\mathrm{d}S_t = \mu_{\text{post}} S_t \mathrm{d}t + \sigma_{\text{post}} S_t \mathrm{d}W_t')
            st.write(f"Parameters: $\\mu = {sp_posterior.process_params['mu']:.4f}$, $\\sigma = {sp_posterior.process_params['sigma']:.4f}$")
        
        # Generate simulations
        with st.spinner("Generating simulations..."):
            prior_sims = sp_prior.simulate()
            posterior_sims = sp_posterior.simulate()
            
            # Get data for plotting
            if 'Close' in data.columns:
                prices = data['Close'].values
            elif 'close' in data.columns:
                prices = data['close'].values
            else:
                prices = None
                st.error("Data does not contain 'Close' or 'close' column.")
                
        if prices is not None:
            # Create three subplots: prior, data, posterior
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Prior simulations
            for path in prior_sims:
                axs[0].plot(range(len(path)), path, alpha=0.6, linewidth=0.8)
            axs[0].set_title("Prior Process Simulations (20 paths)", fontsize=14, fontweight="bold")
            axs[0].set_ylabel("Value", fontsize=11)
            axs[0].grid(True, alpha=0.3)
            
            # Data
            axs[1].plot(range(len(prices)), prices, color='black', linewidth=2, label='Observed Data')
            axs[1].set_title("Imported Data", fontsize=14, fontweight="bold")
            axs[1].set_ylabel("Value", fontsize=11)
            axs[1].grid(True, alpha=0.3)
            
            # Posterior simulations
            for path in posterior_sims:
                axs[2].plot(range(len(path)), path, alpha=0.6, linewidth=0.8)
            axs[2].set_title("Posterior Process Simulations (20 paths)", fontsize=14, fontweight="bold")
            axs[2].set_xlabel("Time step", fontsize=11)
            axs[2].set_ylabel("Value", fontsize=11)
            axs[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Cannot plot because data lacks required columns.")
    else:
        st.info("To view Bayesian posterior results:\n1. Upload data in the 'Data Import' tab\n2. Ensure process type is GBM\n3. Click 'Derive Bayesian Posterior Process'")

# Footer
st.divider()
st.caption("Stochastic Process Simulator - Built with Streamlit and StochasticProcess class")
