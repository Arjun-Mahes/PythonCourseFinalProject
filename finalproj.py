import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, gamma, beta, lognorm, weibull_min, chi2, t, logistic

st.title("Statistical Distribution Fitting Tool")

DIST_MAP = {
    "Normal": norm,
    "Uniform": uniform,
    "Exponential": expon,
    "Gamma": gamma,
    "Beta": beta,
    "Lognormal": lognorm,
    "Weibull": weibull_min,
    "Chi-Square": chi2,
    "Student's t": t,
    "Logistic": logistic
}

# Store the data in a session state (so we don't lose it)
if 'data' not in st.session_state:
    st.session_state.data = None

tab1, tab2 = st.tabs(["Data & Distribution", "Visualization & Accuracy"])

with tab1:
    st.header("Data")

    data_option = st.selectbox("Choose data source:", ["Manual Entry", "CSV Upload", "Generate Normal Distribution"])

    if data_option == "Manual Entry":
        manual_input = st.text_area("Enter values (comma separated)", height=150, placeholder="1.5, 2.3, 3.1, 4.2")
        if manual_input:
            try:
                values = [float(x.strip()) for x in manual_input.split(',') if x.strip()]
                st.session_state.data = np.array(values)
                if len(st.session_state.data) > 0:
                    st.success(f"Loaded {len(st.session_state.data)} points")
            except ValueError:
                st.error("Invalid input")

    elif data_option == "CSV Upload":
        uploaded_file = st.file_uploader("Choose CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))

            col = st.selectbox("Select column", df.columns) if len(df.columns) > 1 else df.columns[0]

            st.session_state.data = df[col].dropna().values
            if len(st.session_state.data) > 0:
                st.success(f"Loaded {len(st.session_state.data)} points | Mean: {np.mean(st.session_state.data):.2f}")

    else:  # Generate Normal Distribution
        mean_param = st.number_input("Mean", value=0.0)
        std_param = st.number_input("Standard Deviation", value=1.0, min_value=0.1)

        if st.button("Generate Data"):
            st.session_state.data = norm.rvs(loc=mean_param, scale=std_param, size=500)

            # Compute stats once and cache
            data_mean = np.mean(st.session_state.data)
            data_std = np.std(st.session_state.data)

            st.write(f"Mean: {data_mean:.4f}")
            st.write(f"Standard Deviation: {data_std:.4f}")
            st.write(f"Variance: {data_std**2:.4f}")  # More efficient than calling np.var

            st.success("Generated 500 data points from normal distribution")

    if st.session_state.data is not None and len(st.session_state.data) > 0:
        st.info(f"Ready: {len(st.session_state.data)} data points")
    else:
        st.warning("Enter data to proceed")

    st.divider()
    st.header("Distribution")

    distribution = st.selectbox("Choose distribution:", list(DIST_MAP.keys()))

with tab2:
    st.header("Visualization")

    if st.session_state.data is not None and len(st.session_state.data) > 0:
        # Cache data reference for efficiency
        data = st.session_state.data

        # Get selected distribution from mapping
        selected_dist = DIST_MAP[distribution]
        params = selected_dist.fit(data)

        fitting_mode = st.radio("Fitting mode:", ["Automatic", "Manual"], horizontal=True)

        if fitting_mode == "Manual":
            # Compute data statistics once
            data_min = data.min()
            data_max = data.max()
            data_std = data.std()
            data_range = data_max - data_min

            # Pre-compute common slider bounds
            loc_min = float(data_min - 2 * data_std)
            loc_max = float(data_max + 2 * data_std)
            scale_max = float(5 * data_std)

            if distribution == "Normal":
                p1 = st.slider("Mean", loc_min, loc_max, float(params[0]))
                p2 = st.slider("STD", 0.01, scale_max, float(params[1]))
                params = (p1, p2)
            elif distribution in ["Gamma", "Lognormal", "Weibull", "Chi-Square", "Student's t"]:
                p1 = st.slider("Shape/DF", 0.1, 20.0, float(params[0]))
                p2 = st.slider("Location", loc_min, loc_max, float(params[1]))
                p3 = st.slider("Scale", 0.01, scale_max, float(params[2]))
                params = (p1, p2, p3)
            elif distribution == "Beta":
                p1 = st.slider("a", 0.1, 20.0, float(params[0]))
                p2 = st.slider("b", 0.1, 20.0, float(params[1]))
                p3 = st.slider("Location", loc_min, loc_max, float(params[2]))
                p4 = st.slider("Scale", 0.01, float(max(data_range, data_std) * 5), float(params[3]))
                params = (p1, p2, p3, p4)
            else:
                p1 = st.slider("Location", loc_min, loc_max, float(params[0]))
                p2 = st.slider("Scale", 0.01, scale_max, float(params[1]))
                params = (p1, p2)

        # Dynamically determine x-axis range to keep fitted distribution visible
        data_std = data.std()
        x_min = data.min() - 0.5 * data_std
        x_max = data.max() + 0.5 * data_std

        # Expand range based on distribution parameters to ensure fitted curve stays visible
        if len(params) >= 2:
            # Extract loc and scale efficiently based on parameter structure
            if len(params) == 2:  # loc, scale
                loc, scale = params
            elif len(params) == 3:  # shape, loc, scale
                loc, scale = params[1], params[2]
            else:  # e.g., beta: a, b, loc, scale
                loc, scale = params[-2], params[-1]

            # Ensure distribution's significant range is included
            x_min = min(x_min, loc - 4 * abs(scale))
            x_max = max(x_max, loc + 4 * abs(scale))

        # Create beautiful professional plot
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#f8f9fa')

        # Plot histogram with gradient-like appearance
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.75,
                                     color='#3498db', edgecolor='#2c3e50',
                                     linewidth=1.5, label='Observed Data')

        # Add gradient effect to histogram bars
        for i, patch in enumerate(patches):
            patch.set_facecolor(plt.cm.Blues(0.4 + 0.4 * (i / len(patches))))

        x = np.linspace(x_min, x_max, 1000)
        pdf = selected_dist.pdf(x, *params)

        # Filter out invalid PDF values (NaN, inf)
        valid_mask = np.isfinite(pdf)

        # Plot fitted distribution with shadow effect
        ax.plot(x[valid_mask], pdf[valid_mask], color='#e74c3c', linewidth=3.5,
                label=f'{fitting_mode} Fit', zorder=5)
        ax.plot(x[valid_mask], pdf[valid_mask], color='#c0392b', linewidth=5,
                alpha=0.3, zorder=4)  # Shadow/glow effect

        # Enhanced labels and title
        ax.set_xlabel('Value', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_title(f'{distribution} Distribution - {fitting_mode} Fit',
                     fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

        # Professional legend
        legend = ax.legend(loc='best', frameon=True, shadow=True, fancybox=True,
                          fontsize=11, framealpha=0.95)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_edgecolor('#bdc3c7')
        legend.get_frame().set_linewidth(1.5)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('#7f8c8d')
        ax.spines['bottom'].set_color('#7f8c8d')

        # Subtle grid on y-axis only
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7', linewidth=0.8)
        ax.set_axisbelow(True)  # Grid behind plot elements

        # Adjust layout for better spacing
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)  

        st.divider()
        st.header("Accuracy")

        hist_counts, bin_edges = np.histogram(data, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5  # Slightly faster than division by 2
        fitted_values = selected_dist.pdf(bin_centers, *params)
        errors = np.abs(hist_counts - fitted_values)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Error", f"{np.mean(errors):.4f}")
        with col2:
            st.metric("Maximum Error", f"{np.max(errors):.4f}")

    else:
        st.warning("No data to visualize. Please enter data first.")
