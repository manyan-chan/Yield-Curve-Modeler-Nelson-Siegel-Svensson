import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize


# --- Nelson-Siegel-Svensson (NSS) Model ---
def nss_yield_curve(maturities, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Calculates yields based on the Nelson-Siegel-Svensson model.
    maturities: array of maturities in years
    beta0: long-term level
    beta1: short-term slope (negative for normal curve)
    beta2: medium-term curvature
    beta3: second curvature component
    tau1: decay factor for slope and first curvature
    tau2: decay factor for second curvature
    """
    m_over_tau1 = maturities / tau1
    m_over_tau2 = maturities / tau2

    term1 = beta0
    term2 = beta1 * (1 - np.exp(-m_over_tau1)) / m_over_tau1
    term3 = beta2 * ((1 - np.exp(-m_over_tau1)) / m_over_tau1 - np.exp(-m_over_tau1))
    term4 = beta3 * ((1 - np.exp(-m_over_tau2)) / m_over_tau2 - np.exp(-m_over_tau2))

    return term1 + term2 + term3 + term4


def nss_objective_function(params, maturities, market_yields):
    """
    Objective function to minimize for NSS fitting (sum of squared errors).
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    fitted_yields = nss_yield_curve(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
    error = np.sum((fitted_yields - market_yields) ** 2)
    return error


# --- Bond Pricing Helper ---
def price_bond_from_nss(
    face_value, coupon_rate, years_to_maturity, nss_params, payments_per_year=2
):
    """
    Prices a bond using a fitted NSS yield curve to get spot rates.
    Assumes coupon payments are made at the specified frequency.
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = nss_params

    num_payments = int(years_to_maturity * payments_per_year)
    coupon_payment = (coupon_rate / payments_per_year) * face_value

    payment_times = np.array([(i + 1) / payments_per_year for i in range(num_payments)])

    # Get spot rates from NSS curve for each payment time
    spot_rates = (
        nss_yield_curve(payment_times, beta0, beta1, beta2, beta3, tau1, tau2) / 100
    )  # Convert % to decimal

    pv_coupons = 0
    for i in range(num_payments - 1):  # All coupons except the last one
        pv_coupons += coupon_payment / (
            (1 + spot_rates[i] / payments_per_year) ** (i + 1)
        )

    # Last payment includes final coupon and principal
    pv_last_payment = (coupon_payment + face_value) / (
        (1 + spot_rates[num_payments - 1] / payments_per_year) ** num_payments
    )

    bond_price = pv_coupons + pv_last_payment
    return bond_price


# --- Example Market Data (US Treasury Yields - Replace with more current/dynamic data if desired) ---
# Maturities in years, Yields in %
EXAMPLE_MATURITIES = np.array(
    [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
)  # 3m, 6m, 1y, 2y, ... 30y
EXAMPLE_YIELDS = np.array(
    [5.30, 5.40, 5.10, 4.80, 4.60, 4.50, 4.55, 4.50, 4.70, 4.65]
)  # Example data

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Interactive Yield Curve Modeler (Nelson-Siegel-Svensson)")
st.markdown("""
This application demonstrates fitting the Nelson-Siegel-Svensson (NSS) model to observed market yields.
You can interact with the NSS parameters to see their effect on the curve shape or fit the model to example data.
Optionally, price a bond using the fitted yield curve.
""")

# --- Sidebar for Controls ---
st.sidebar.header("NSS Model Controls")
control_mode = st.sidebar.radio(
    "Control Mode:", ("Interactive Sliders", "Fit to Market Data"), index=1
)

st.sidebar.subheader("Observed Market Yields")
st.sidebar.markdown("_(You can edit these example values)_")

# Editable market data in the sidebar
cols = st.sidebar.columns(2)
edited_maturities_str = cols[0].text_area(
    "Maturities (years, comma-separated):",
    value=", ".join(map(str, EXAMPLE_MATURITIES)),
    height=150,
)
edited_yields_str = cols[1].text_area(
    "Yields (%, comma-separated):",
    value=", ".join(map(str, EXAMPLE_YIELDS)),
    height=150,
)

try:
    market_maturities = np.array(
        [float(x.strip()) for x in edited_maturities_str.split(",") if x.strip()]
    )
    market_yields = np.array(
        [float(x.strip()) for x in edited_yields_str.split(",") if x.strip()]
    )
    if len(market_maturities) != len(market_yields):
        st.sidebar.error("Number of maturities and yields must match.")
        st.stop()
    if not all(market_maturities > 0) or not all(np.isfinite(market_yields)):
        st.sidebar.error("Maturities must be positive. Yields must be finite numbers.")
        st.stop()
except ValueError:
    st.sidebar.error(
        "Invalid input for maturities or yields. Please use comma-separated numbers."
    )
    st.stop()


# NSS Parameters
# Initial guesses and bounds for optimization, and also for sliders
initial_params = {
    "beta0": 3.0,  # Long-term level %
    "beta1": -1.5,  # Initial slope %
    "beta2": -1.0,  # First curvature %
    "beta3": 0.5,  # Second curvature %
    "tau1": 1.5,  # Decay for beta1, beta2 (years)
    "tau2": 5.0,  # Decay for beta3 (years)
}

param_bounds = {
    "beta0": (0, 15),
    "beta1": (-10, 10),
    "beta2": (-10, 10),
    "beta3": (-10, 10),
    "tau1": (0.1, 10),  # Must be > 0
    "tau2": (0.1, 30),  # Must be > 0, typically tau2 > tau1
}

# --- Main Panel ---
col1, col2 = st.columns([2, 1])  # Plot on left, parameters/results on right

with col2:
    st.subheader("NSS Parameters")
    if control_mode == "Interactive Sliders":
        beta0_s = st.slider(
            "β₀ (Level)",
            float(param_bounds["beta0"][0]),
            float(param_bounds["beta0"][1]),
            float(initial_params["beta0"]),
            0.1,
        )
        beta1_s = st.slider(
            "β₁ (Slope)",
            float(param_bounds["beta1"][0]),
            float(param_bounds["beta1"][1]),
            float(initial_params["beta1"]),
            0.1,
        )
        beta2_s = st.slider(
            "β₂ (Curvature 1)",
            float(param_bounds["beta2"][0]),
            float(param_bounds["beta2"][1]),
            float(initial_params["beta2"]),
            0.1,
        )
        beta3_s = st.slider(
            "β₃ (Curvature 2)",
            float(param_bounds["beta3"][0]),
            float(param_bounds["beta3"][1]),
            float(initial_params["beta3"]),
            0.1,
        )
        tau1_s = st.slider(
            "τ₁ (Decay 1)",
            float(param_bounds["tau1"][0]),
            float(param_bounds["tau1"][1]),
            float(initial_params["tau1"]),
            0.1,
        )
        tau2_s = st.slider(
            "τ₂ (Decay 2)",
            float(param_bounds["tau2"][0]),
            float(param_bounds["tau2"][1]),
            float(initial_params["tau2"]),
            0.1,
        )

        # Ensure tau1 != tau2 for the model to be well-defined if beta3 is non-zero
        if abs(tau1_s - tau2_s) < 0.01 and abs(beta3_s) > 0.01:  # A small tolerance
            st.warning(
                "τ₁ and τ₂ are very close. This can make the model less stable or parameters unidentifiable if β₃ is non-zero."
            )
            # Potentially auto-adjust one slightly if they become equal
            # if tau1_s == tau2_s: tau2_s += 0.01

        current_params_display = [beta0_s, beta1_s, beta2_s, beta3_s, tau1_s, tau2_s]
        param_names_display = ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"]

    elif control_mode == "Fit to Market Data":
        st.markdown("Fitting NSS model to the provided market yields...")

        # Convert dicts to lists for optimizer
        p0 = [
            initial_params[k]
            for k in ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
        ]
        bnds = [
            param_bounds[k]
            for k in ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
        ]

        # Constraint: tau1 != tau2 (or handle gracefully if they converge)
        # A simple penalty approach or a transformation can be used, but for `minimize`
        # often it's about good bounds and initial guesses.
        # Here we will rely on bounds and the optimizer to hopefully find a good spot.
        # A more robust solution might use a constrained optimizer that directly handles tau1 != tau2.

        # Ensure tau1 != tau2 initially for fit if they happen to be the same.
        if abs(p0[4] - p0[5]) < 0.01:
            p0[5] += 0.1  # Slightly perturb tau2 if too close to tau1

        result = minimize(
            nss_objective_function,
            p0,
            args=(market_maturities, market_yields),
            method="L-BFGS-B",  # Good for bounded problems
            bounds=bnds,
            options={"ftol": 1e-9, "gtol": 1e-7, "maxiter": 500},
        )

        if result.success:
            fitted_params = result.x
            st.success("Model fitting successful!")
            param_names_display = [
                "β₀ (fit)",
                "β₁ (fit)",
                "β₂ (fit)",
                "β₃ (fit)",
                "τ₁ (fit)",
                "τ₂ (fit)",
            ]
            current_params_display = fitted_params

            if (
                abs(fitted_params[4] - fitted_params[5]) < 0.05
                and abs(fitted_params[3]) > 0.01
            ):  # Check if taus are too close post-fit
                st.warning(
                    f"Fitted τ₁ ({fitted_params[4]:.2f}) and τ₂ ({fitted_params[5]:.2f}) are very close. "
                    "The β₃ parameter might be unstable or less meaningful."
                )

        else:
            st.error(f"Model fitting failed: {result.message}")
            st.text("Using initial parameters for plot.")
            param_names_display = [
                "β₀ (initial)",
                "β₁ (initial)",
                "β₂ (initial)",
                "β₃ (initial)",
                "τ₁ (initial)",
                "τ₂ (initial)",
            ]
            current_params_display = p0

    # Display parameters
    for name, val in zip(param_names_display, current_params_display):
        st.metric(label=name, value=f"{val:.3f}")


# Plotting
max_maturity_plot = max(
    30, np.max(market_maturities) * 1.1
)  # Extend plot beyond max input maturity
plot_maturities = np.linspace(
    0.01, max_maturity_plot, 200
)  # Dense maturities for smooth curve

beta0_p, beta1_p, beta2_p, beta3_p, tau1_p, tau2_p = current_params_display
fitted_yields_plot = nss_yield_curve(
    plot_maturities, beta0_p, beta1_p, beta2_p, beta3_p, tau1_p, tau2_p
)

# Create plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=market_maturities,
        y=market_yields,
        mode="markers",
        name="Market Yields",
        marker=dict(size=8, color="blue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_maturities,
        y=fitted_yields_plot,
        mode="lines",
        name="Fitted NSS Curve",
        line=dict(color="red", width=2),
    )
)

fig.update_layout(
    title="Yield Curve: Market vs. Fitted NSS Model",
    xaxis_title="Maturity (Years)",
    yaxis_title="Yield (%)",
    legend_title_text="Data Series",
    height=500,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightGrey")
fig.update_xaxes(
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor="LightGrey",
    range=[0, max_maturity_plot],
)

with col1:
    st.plotly_chart(fig, use_container_width=True)

    # Display component contributions (optional, for understanding)
    if st.checkbox("Show NSS Component Contributions to Yield Curve", value=False):
        term1_plot = np.full_like(plot_maturities, beta0_p)
        term2_plot = (
            beta1_p
            * (1 - np.exp(-plot_maturities / tau1_p))
            / (plot_maturities / tau1_p)
        )
        term3_plot = beta2_p * (
            (1 - np.exp(-plot_maturities / tau1_p)) / (plot_maturities / tau1_p)
            - np.exp(-plot_maturities / tau1_p)
        )
        term4_plot = beta3_p * (
            (1 - np.exp(-plot_maturities / tau2_p)) / (plot_maturities / tau2_p)
            - np.exp(-plot_maturities / tau2_p)
        )

        fig_components = go.Figure()
        fig_components.add_trace(
            go.Scatter(x=plot_maturities, y=term1_plot, mode="lines", name="β₀ (Level)")
        )
        fig_components.add_trace(
            go.Scatter(
                x=plot_maturities, y=term2_plot, mode="lines", name="β₁ Term (Slope)"
            )
        )
        fig_components.add_trace(
            go.Scatter(
                x=plot_maturities,
                y=term3_plot,
                mode="lines",
                name="β₂ Term (Curvature 1)",
            )
        )
        fig_components.add_trace(
            go.Scatter(
                x=plot_maturities,
                y=term4_plot,
                mode="lines",
                name="β₃ Term (Curvature 2)",
            )
        )
        fig_components.add_trace(
            go.Scatter(
                x=plot_maturities,
                y=fitted_yields_plot,
                mode="lines",
                name="Total Fitted NSS Curve",
                line=dict(color="black", dash="dash"),
            )
        )

        fig_components.update_layout(
            title="NSS Model Component Contributions",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield Contribution (%)",
            height=400,
        )
        st.plotly_chart(fig_components, use_container_width=True)

# --- Optional: Bond Pricing Section ---
st.sidebar.markdown("---")
st.sidebar.header("Bond Pricer (Optional)")
price_this_bond = st.sidebar.checkbox("Enable Bond Pricing")

if price_this_bond:
    st.sidebar.subheader("Bond Characteristics")
    face_val = st.sidebar.number_input(
        "Face Value ($)", min_value=0.0, value=1000.0, step=100.0
    )
    coupon_r = (
        st.sidebar.number_input(
            "Annual Coupon Rate (%)", min_value=0.0, value=5.0, step=0.1
        )
        / 100
    )
    years_mat = st.sidebar.number_input(
        "Years to Maturity", min_value=0.1, value=10.0, step=0.5
    )
    payments_yr = st.sidebar.selectbox("Payments per Year", [1, 2, 4], index=1)

    if st.sidebar.button("Price Bond using Fitted Curve"):
        if "current_params_display" not in locals() or current_params_display is None:
            st.sidebar.error(
                "NSS parameters not available. Please fit or set parameters first."
            )
        else:
            try:
                bond_price = price_bond_from_nss(
                    face_val, coupon_r, years_mat, current_params_display, payments_yr
                )
                st.sidebar.success(f"Calculated Bond Price: ${bond_price:.2f}")
            except Exception as e:
                st.sidebar.error(f"Error pricing bond: {e}")
                st.sidebar.text(
                    "Ensure maturities for payments are within the reliable range of the fitted curve."
                )


# --- Explanation Section ---
st.markdown("---")
st.header("Understanding the Nelson-Siegel-Svensson Model")
st.markdown(r"""
The Nelson-Siegel-Svensson (NSS) model describes the yield $y(m)$ for a given maturity $m$ as:
$y(m) = \beta_0 + \beta_1 \left( \frac{1 - e^{-m/\tau_1}}{m/\tau_1} \right) + \beta_2 \left( \frac{1 - e^{-m/\tau_1}}{m/\tau_1} - e^{-m/\tau_1} \right) + \beta_3 \left( \frac{1 - e^{-m/\tau_2}}{m/\tau_2} - e^{-m/\tau_2} \right)$

Where:
- $\beta_0$: Represents the long-term level of interest rates (as $m \to \infty$, $y(m) \to \beta_0$).
- $\beta_1$: Influences the short-term slope. A negative $\beta_1$ typically corresponds to an upward-sloping (normal) yield curve at the short end. As $m \to 0$, $y(m) \to \beta_0 + \beta_1$.
- $\beta_2$: Introduces a hump or trough (curvature) in the medium-term part of the curve. Its loading factor peaks at an intermediate maturity.
- $\beta_3$: Introduces a second hump or trough, allowing for more complex curve shapes, especially at longer maturities.
- $\tau_1$: A time constant that determines the decay rate of the $\beta_1$ and $\beta_2$ components. Smaller $\tau_1$ means faster decay (effects are concentrated at shorter maturities).
- $\tau_2$: A time constant that determines the decay rate of the $\beta_3$ component. Smaller $\tau_2$ means faster decay for this second curvature.

**Model Fitting:** The application uses optimization (`scipy.optimize.minimize` with L-BFGS-B) to find the NSS parameters $(\beta_0, \beta_1, \beta_2, \beta_3, \tau_1, \tau_2)$ that minimize the sum of squared differences between the model's calculated yields and the observed market yields.

**Bond Pricing:** The fitted NSS curve provides spot rates for any maturity. A bond's price is the sum of its discounted future cash flows (coupons and principal), where each cash flow is discounted using the spot rate corresponding to its payment time.
""")
