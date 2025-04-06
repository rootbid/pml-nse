def load_and_forecast(df, model_names=None, forecast_days=10):
    """Load saved models and generate forecasts."""
    if model_names is None:
        model_names = [
            "Bayesian Linear",
            "AR(1)",
            "Hierarchical",
            "GPR",
            "Dynamic Regression",
        ]

    results = {}

    for model_name in model_names:
        logging.info(f"Loading and forecasting with {model_name} model...")
        try:
            # Load saved model data
            model_data = load_trained_model(model_name)
            if model_data is None:
                logging.error(f"Could not load {model_name} model data")
                continue

            # Generate forecasts directly using trace and scalers
            if model_name == "Bayesian Linear":
                forecast_dates, pred_mean, pred_lower, pred_upper = (
                    forecast_bayesian_linear(df, model_data, forecast_days)
                )
            elif model_name == "Dynamic Regression":
                forecast_dates, pred_mean, pred_lower, pred_upper = (
                    forecast_dynamic_regression(df, model_data, forecast_days)
                )
            elif model_name == "AR(1)":
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_ar1(
                    df, model_data, forecast_days
                )
            elif model_name == "Hierarchical":
                forecast_dates, pred_mean, pred_lower, pred_upper = (
                    forecast_hierarchical(df, model_data, forecast_days)
                )
            elif model_name == "GPR":
                forecast_dates, pred_mean, pred_lower, pred_upper = forecast_gpr(
                    df, model_data, forecast_days
                )

            # Plot forecasts
            plot_forecast(
                df, forecast_dates, pred_mean, pred_lower, pred_upper, model_name
            )
            results[model_name] = {
                "dates": forecast_dates,
                "mean": pred_mean,
                "lower": pred_lower,
                "upper": pred_upper,
            }

        except Exception as e:
            logging.error(f"Error forecasting with {model_name} model: {str(e)}")
            traceback.print_exc()
            continue

    return results


# Usage example
if __name__ == "__main__":
    # Load and process data
    df = load_nifty_data("data/nifty_data.csv")

    # Load models and generate forecasts
    forecast_results = load_and_forecast(df)
