from flask import Flask, render_template, request, url_for, session
from flask_session import Session
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session' 
Session(app)


def generate_data(N, mu, beta0, beta1, sigma2, S):
    """
    Generate data for linear regression simulation with specified parameters.
    """
    # Generate initial dataset
    X = np.random.uniform(0, 1, N)
    
    # Generate Y values using the true model: Y = beta0 + beta1*X + error
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error
    
    # Reshape X for sklearn
    X_reshaped = X.reshape(-1, 1)
    
    # Fit initial linear regression model
    model = LinearRegression()
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Generate initial scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X, model.predict(X_reshaped), color='red', linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Regression (y = {slope:.2f}x + {intercept:.2f})")
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()
    
    # Run S simulations
    slopes = []
    intercepts = []
    
    for _ in range(S):
        # Generate simulated dataset using the same model
        X_sim = np.random.uniform(0, 1, N)
        X_sim_reshaped = X_sim.reshape(-1, 1)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim
        
        # Fit model to simulated data
        sim_model = LinearRegression()
        sim_model.fit(X_sim_reshaped, Y_sim)
        
        # Store results
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)
    
    # Generate histogram of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()
    
    # Calculate proportions of more extreme values
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_extreme = sum(i < intercept for i in intercepts) / S
    
    return (
        X, Y, slope, intercept, plot1_path, plot2_path,
        slope_more_extreme, intercept_extreme, slopes, intercepts
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X, Y, slope, intercept, plot1, plot2,
            slope_extreme, intercept_extreme, slopes, intercepts
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    else:
    # Initialize variables to empty strings for GET request
        return render_template(
            "index.html",
            N='',
            mu='',
            sigma2='',
            beta0='',
            beta1='',
            S='',
        )

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == "two_sided":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= 
                         np.abs(observed_stat - hypothesized_value))
    elif test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:  # less
        p_value = np.mean(simulated_stats <= observed_stat)

    # Fun message for very small p-values
    if p_value <= 0.0001:
        fun_message = "Wow! This is one ₛₘₐₗₗ p-value."
    else:
        fun_message = None

    # Plot histogram with test results
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, density=True, alpha=0.7,
             label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--',
                label=f'Observed Statistic: {observed_stat:.3f}')
    plt.axvline(hypothesized_value, color='green', linestyle='-',
                label=f'Hypothesized Value: {hypothesized_value:.3f}')
    plt.title(f'Hypothesis Test Results (p-value: {p_value:.4f})')
    plt.xlabel('Statistic Value')
    plt.ylabel('Density')
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate confidence interval
    alpha = 1 - confidence_level
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    
    # Using t-distribution for confidence interval
    t_value = stats.t.ppf(1 - alpha/2, N-2)
    ci_lower = mean_estimate - t_value * std_estimate
    ci_upper = mean_estimate + t_value * std_estimate
    
    # Check if true parameter is in confidence interval
    includes_true = ci_lower <= true_param <= ci_upper

    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot individual estimates
    plt.plot(estimates, np.random.normal(0, 0.1, len(estimates)), 'o', 
             alpha=0.2, color='gray', label='Individual Estimates')
    
    # Plot mean estimate with color based on true parameter inclusion
    point_color = 'green' if includes_true else 'red'
    plt.plot(mean_estimate, 0, 'o', color=point_color, markersize=10,
             label='Mean Estimate')
    
    # Plot confidence interval
    plt.hlines(y=0, xmin=ci_lower, xmax=ci_upper, color='blue',
               label=f'{confidence_level*100}% Confidence Interval')
    
    # Plot true parameter
    plt.axvline(x=true_param, color='black', linestyle='--',
                label='True Parameter')
    
    plt.title(f'Confidence Interval for {parameter.capitalize()}')
    plt.xlabel('Parameter Value')
    plt.legend()
    
    # Remove y-axis as it's just for visualization
    plt.gca().set_yticks([])
    
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True)