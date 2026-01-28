import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# Page config
# =========================
st.set_page_config(page_title="Bed Elevation Trend & Projection", layout="wide")
st.title("🪨 Bed Elevation Trend Analysis & 25-Year Projection")

# =========================
# User inputs
# =========================
start_time_str = st.text_input("Start time (yyyy-mm-dd hh:00)", "2025-01-01 00:00")
dt_hours = st.number_input("Time interval (hours)", min_value=1, value=1)
raw_data = st.text_area("Paste bed elevation data (one value per line)", height=250)
run = st.button("Run Analysis")

# =========================
# Helper functions
# =========================
def parse_data(text):
    return np.array([float(x) for x in text.splitlines() if x.strip() != ""], dtype=float)

def exp_saturation(t, a, b, c):
    with np.errstate(over='ignore', invalid='ignore'):
        return a * (1 - np.exp(-b * t)) + c

def exp_decay(t, a, b, c):
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.exp(-b * t) + c

def asymptotic(t, a, b, c):
    with np.errstate(divide='ignore', invalid='ignore'):
        return a / (1 + b * t) + c

def logistic(t, L, k, t0):
    with np.errstate(over='ignore', invalid='ignore'):
        return L / (1 + np.exp(-k * (t - t0)))

def gompertz(t, a, b, c):
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.exp(-b * np.exp(-c * t))

def power_law(t, a, b, c):
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.power(t + 1, b) + c

def safe_fit(func, x, y, initial_guess=None, name="Model"):
    """
    Safely attempt to fit a model with multiple strategies
    """
    try:
        # Try with default parameters
        if initial_guess is not None:
            popt, _ = curve_fit(func, x, y, p0=initial_guess, maxfev=10000)
        else:
            popt, _ = curve_fit(func, x, y, maxfev=10000)
        
        # Test if the fit produces valid values
        y_pred = func(x, *popt)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return None, None
        
        return popt, y_pred
    except:
        try:
            # Try with looser bounds
            n_params = func.__code__.co_argcount - 1
            if initial_guess is not None:
                popt, _ = curve_fit(func, x, y, p0=initial_guess, 
                                   bounds=([-1e10]*n_params, [1e10]*n_params),
                                   maxfev=5000)
            else:
                popt, _ = curve_fit(func, x, y, 
                                   bounds=([-1e10]*n_params, [1e10]*n_params),
                                   maxfev=5000)
            
            y_pred = func(x, *popt)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return None, None
            
            return popt, y_pred
        except:
            return None, None

def get_initial_guesses(x, y):
    """
    Generate reasonable initial guesses based on data characteristics
    """
    y_range = np.max(y) - np.min(y)
    y_mean = np.mean(y)
    y_start = y[0]
    y_end = y[-1]
    
    guesses = {
        'exp_saturation': [y_range, 0.1, y_start],
        'exp_decay': [y_range, 0.1, y_end],
        'asymptotic': [y_range, 0.1, y_mean],
        'logistic': [y_mean, 0.1, np.median(x)],
        'gompertz': [np.max(y), 0.1, 0.1],
        'power_law': [y_start, 0.1, 0]
    }
    
    return guesses

# =========================
# Main logic
# =========================
if run:
    with st.spinner('Analyzing data... This may take a moment.'):
        try:
            data = parse_data(raw_data)
            n = len(data)
            if n < 6:
                raise ValueError("At least 6 data points are required.")

            t0 = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
            hours = np.arange(n) * float(dt_hours)
            years = hours / (24 * 365.25)

            models = {}
            equations = {}
            params = {}
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get initial guesses
            guesses = get_initial_guesses(years, data)

            # Linear (always works)
            status_text.text("Fitting Linear model...")
            progress_bar.progress(10)
            try:
                p_lin = np.polyfit(years, data, 1)
                models["Linear"] = np.polyval(p_lin, years)
                equations["Linear"] = f"y = {p_lin[0]:.4f}·t + {p_lin[1]:.4f}"
                params["Linear"] = p_lin
            except:
                pass

            # Polynomial (degree 2)
            status_text.text("Fitting Polynomial model...")
            progress_bar.progress(20)
            try:
                p_poly = np.polyfit(years, data, 2)
                models["Polynomial (degree 2)"] = np.polyval(p_poly, years)
                equations["Polynomial (degree 2)"] = f"y = {p_poly[0]:.4f}·t² + {p_poly[1]:.4f}·t + {p_poly[2]:.4f}"
                params["Polynomial (degree 2)"] = p_poly
            except:
                pass

            # Logarithmic
            status_text.text("Fitting Logarithmic model...")
            progress_bar.progress(30)
            try:
                p_log = np.polyfit(np.log(years + 1), data, 1)
                models["Logarithmic"] = np.polyval(p_log, np.log(years + 1))
                equations["Logarithmic"] = f"y = {p_log[0]:.4f}·ln(t+1) + {p_log[1]:.4f}"
                params["Logarithmic"] = p_log
            except:
                pass

            # Exponential saturation
            status_text.text("Fitting Exponential Saturation model...")
            progress_bar.progress(40)
            popt, y_pred = safe_fit(exp_saturation, years, data, 
                                    initial_guess=guesses['exp_saturation'])
            if popt is not None:
                models["Exponential Saturation"] = y_pred
                equations["Exponential Saturation"] = f"y = {popt[0]:.4f}(1-e^(-{popt[1]:.4f}t))+{popt[2]:.4f}"
                params["Exponential Saturation"] = popt

            # Exponential decay
            status_text.text("Fitting Exponential Decay model...")
            progress_bar.progress(50)
            popt, y_pred = safe_fit(exp_decay, years, data,
                                    initial_guess=guesses['exp_decay'])
            if popt is not None:
                models["Exponential Decay"] = y_pred
                equations["Exponential Decay"] = f"y = {popt[0]:.4f}e^(-{popt[1]:.4f}t)+{popt[2]:.4f}"
                params["Exponential Decay"] = popt

            # Asymptotic
            status_text.text("Fitting Asymptotic model...")
            progress_bar.progress(60)
            popt, y_pred = safe_fit(asymptotic, years, data,
                                    initial_guess=guesses['asymptotic'])
            if popt is not None:
                models["Asymptotic"] = y_pred
                equations["Asymptotic"] = f"y = {popt[0]:.4f}/(1+{popt[1]:.4f}t)+{popt[2]:.4f}"
                params["Asymptotic"] = popt

            # Logistic
            status_text.text("Fitting Logistic model...")
            progress_bar.progress(70)
            popt, y_pred = safe_fit(logistic, years, data,
                                    initial_guess=guesses['logistic'])
            if popt is not None:
                models["Logistic"] = y_pred
                equations["Logistic"] = f"y = {popt[0]:.4f}/(1+e^(-{popt[1]:.4f}(t-{popt[2]:.4f})))"
                params["Logistic"] = popt

            # Gompertz
            status_text.text("Fitting Gompertz model...")
            progress_bar.progress(80)
            popt, y_pred = safe_fit(gompertz, years, data,
                                    initial_guess=guesses['gompertz'])
            if popt is not None:
                models["Gompertz"] = y_pred
                equations["Gompertz"] = f"y = {popt[0]:.4f}e^(-{popt[1]:.4f}e^(-{popt[2]:.4f}t))"
                params["Gompertz"] = popt

            # Power-law
            status_text.text("Fitting Power Law model...")
            progress_bar.progress(90)
            popt, y_pred = safe_fit(power_law, years, data,
                                    initial_guess=guesses['power_law'])
            if popt is not None:
                models["Power Law"] = y_pred
                equations["Power Law"] = f"y = {popt[0]:.4f}(t+1)^{popt[1]:.4f}+{popt[2]:.4f}"
                params["Power Law"] = popt

            # Check if we have at least one successful model
            if len(models) == 0:
                raise ValueError("No models could be successfully fitted to the data. The data may be too irregular or contain outliers.")

            # =========================
            # R² table
            # =========================
            status_text.text("Calculating R² scores...")
            r2_rows = []
            for name, y in models.items():
                r2_rows.append({
                    "Model": name,
                    "R²": round(r2_score(data, y), 4)
                })

            r2_df = pd.DataFrame(r2_rows).sort_values("R²", ascending=False).reset_index(drop=True)

            best_model = r2_df.iloc[0]["Model"]
            best_fit_obs = models[best_model]
            best_eq = equations[best_model]
            best_r2 = r2_df.iloc[0]["R²"]
            best_params = params[best_model]

            # =========================
            # Projection (25y) - OPTIMIZED
            # =========================
            status_text.text("Generating 25-year projection...")
            
            # Calculate projection with smart sampling to avoid memory issues
            total_hours = 25 * 365.25 * 24
            
            # For output: sample every dt_hours
            if dt_hours <= 24:
                # For hourly/sub-daily data, sample daily for projection output
                sample_hours = 24
            else:
                sample_hours = dt_hours
            
            total_steps = int(total_hours / sample_hours)
            total_years = np.arange(total_steps) * sample_hours / (24 * 365.25)

            # Define projection functions
            if best_model == "Linear":
                proj = np.polyval(best_params, total_years)
            elif best_model == "Polynomial (degree 2)":
                proj = np.polyval(best_params, total_years)
            elif best_model == "Logarithmic":
                proj = np.polyval(best_params, np.log(total_years + 1))
            elif best_model == "Exponential Saturation":
                proj = exp_saturation(total_years, *best_params)
            elif best_model == "Exponential Decay":
                proj = exp_decay(total_years, *best_params)
            elif best_model == "Asymptotic":
                proj = asymptotic(total_years, *best_params)
            elif best_model == "Logistic":
                proj = logistic(total_years, *best_params)
            elif best_model == "Gompertz":
                proj = gompertz(total_years, *best_params)
            elif best_model == "Power Law":
                proj = power_law(total_years, *best_params)

            # Generate time array for full 25-year period
            # Use original dt_hours interval for the entire projection
            total_steps_original = int(25 * 365.25 * 24 / dt_hours)
            time_full = [(t0 + timedelta(hours=float(i * dt_hours))).strftime("%m/%d/%Y %H:%M")
                         for i in range(total_steps_original)]
            
            # Calculate full projection at original resolution
            total_years_original = np.arange(total_steps_original) * dt_hours / (24 * 365.25)
            
            if best_model == "Linear":
                proj_full = np.polyval(best_params, total_years_original)
            elif best_model == "Polynomial (degree 2)":
                proj_full = np.polyval(best_params, total_years_original)
            elif best_model == "Logarithmic":
                proj_full = np.polyval(best_params, np.log(total_years_original + 1))
            elif best_model == "Exponential Saturation":
                proj_full = exp_saturation(total_years_original, *best_params)
            elif best_model == "Exponential Decay":
                proj_full = exp_decay(total_years_original, *best_params)
            elif best_model == "Asymptotic":
                proj_full = asymptotic(total_years_original, *best_params)
            elif best_model == "Logistic":
                proj_full = logistic(total_years_original, *best_params)
            elif best_model == "Gompertz":
                proj_full = gompertz(total_years_original, *best_params)
            elif best_model == "Power Law":
                proj_full = power_law(total_years_original, *best_params)
            
            # Create data_model column: observed data for first n rows, NaN for the rest
            data_model = np.full(total_steps_original, np.nan)
            data_model[:n] = data
            
            result_df = pd.DataFrame({
                "time": time_full,
                "data model": data_model,
                "best fit": proj_full
            })
            
            # Warn if CSV will be very large
            if total_steps_original > 100000:
                st.session_state.large_csv_warning = f"⚠️ Note: CSV will contain {total_steps_original:,} rows (~{total_steps_original*50//1024//1024}MB). This may take time to download."
            else:
                st.session_state.large_csv_warning = None

            progress_bar.progress(100)
            status_text.text("Done!")
            
            # For graphs, use sampled data for better performance
            sample_years = total_years  # Already calculated earlier with sampling
            sample_proj = proj  # Already calculated earlier with sampling
            
            st.session_state.r2_df = r2_df
            st.session_state.result_df = result_df
            st.session_state.best_model = best_model
            st.session_state.best_r2 = best_r2
            st.session_state.best_eq = best_eq
            st.session_state.years_obs = years
            st.session_state.data_obs = data
            st.session_state.best_fit_obs = best_fit_obs
            st.session_state.years_full = sample_years  # Use sampled for graphs
            st.session_state.proj = sample_proj  # Use sampled for graphs
            st.session_state.n_models_fitted = len(models)
            st.session_state.csv_rows = len(result_df)
            st.session_state.sample_info = f"CSV contains all {len(result_df):,} rows at {dt_hours}-hour intervals. Graphs use optimized sampling for display."

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully fitted {len(models)} models to your data!")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tips: Check that your data is properly formatted (one number per line) and contains at least 6 data points.")

# =========================
# Output
# =========================
if "result_df" in st.session_state:

    st.subheader("📊 R² Comparison (All Models)")
    st.write(f"*{st.session_state.n_models_fitted} models successfully fitted*")
    st.dataframe(st.session_state.r2_df, hide_index=True)

    st.subheader("📐 Best Fitted Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", st.session_state.best_model)
    with col2:
        st.metric("R² Score", f"{st.session_state.best_r2:.4f}")
    with col3:
        st.write("**Equation:**")
        st.code(st.session_state.best_eq)

    # Plot 1: Observed vs Fit
    st.subheader("📈 Observed Data vs Best Fit")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(st.session_state.years_obs, st.session_state.data_obs, 'o-', 
             label="Observed", linewidth=2, markersize=6, color='#1f77b4')
    ax1.plot(st.session_state.years_obs, st.session_state.best_fit_obs, '--', 
             label="Best Fit", linewidth=2, color='#ff7f0e')
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Bed Elevation", fontsize=12)
    ax1.set_title("Observed Data vs Best Fit Model", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Plot 2: 25-year projection
    st.subheader("🔮 25-Year Projection")
    st.info(st.session_state.sample_info)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(st.session_state.years_full, st.session_state.proj, 
             label="Projection", linewidth=2, color='#2ca02c')
    ax2.axvline(st.session_state.years_obs[-1], linestyle="--", 
                color="red", linewidth=2, label="End of Observation", alpha=0.7)
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Bed Elevation", fontsize=12)
    ax2.set_title(f"25-Year Projection using {st.session_state.best_model} Model", 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Download button
    st.subheader("💾 Download Results")
    
    if st.session_state.get('large_csv_warning'):
        st.warning(st.session_state.large_csv_warning)
    
    csv_data = st.session_state.result_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name="bed_elevation_trend_25y.csv",
        mime="text/csv"
    )
    
    st.caption(f"CSV contains {st.session_state.csv_rows:,} rows (25 years at {dt_hours}-hour intervals)")
    st.caption("• 'data model' column: observed data only (rest is empty)")
    st.caption("• 'best fit' column: continuous projection for all 25 years")
