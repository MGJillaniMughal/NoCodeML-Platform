import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.time_series import TSForecastingExperiment
from pycaret.clustering import ClusteringExperiment
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import date
import warnings
import shap
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Constants for file paths and export directories
DATA_FILE = 'dataset.csv'
PROCESSED_DATA_FILE = 'processed_dataset.csv'
EXPORT_PATH = 'exports/charts/'
MODEL_FILE_BASE = f"best_model_{date.today().strftime('%m-%d-%Y')}"
EXPORT_DATA_FILE = f"exported_data_{date.today().strftime('%m-%d-%Y')}.csv"

# Ensure the export directory exists
os.makedirs(EXPORT_PATH, exist_ok=True)

# Initialize session state variables
st.session_state.setdefault('data_uploaded', False)
st.session_state.setdefault('processed_data', False)
st.session_state.setdefault('model_file', None)

# Set the page configuration
st.set_page_config(layout="wide", page_title="Low Code No Code ML App", page_icon="🤖")


def setup_sidebar():
    """Setup the sidebar with branding and navigation."""
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Low Code No Code Auto ML App")
        st.info("Developed By: Jillani Soft Tech 😎")
        # Sidebar styling and links
        st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #2c3e50;
            }
            </style>
            """, unsafe_allow_html=True)
        # Social Media Links
        st.markdown("""
            <div style="display: flex; justify-content: space-evenly;">
                <a href="https://github.com/MGJillaniMughal" target="_blank"><img src="https://img.icons8.com/?size=35&id=63777&format=png&color=000000"/></a>
                <a href="https://www.linkedin.com/in/jillanisofttech/" target="_blank"><img src="https://img.icons8.com/?size=35&id=xuvGCOXi8Wyg&format=png&color=000000"/></a>
                <a href="https://www.kaggle.com/jillanisofttech" target="_blank"><img src="https://img.icons8.com/?size=30&id=Omk4fWoSmCHm&format=png&color=000000"/></a>
                <a href="https://jillanisofttech.medium.com/" target="_blank"><img src="https://img.icons8.com/?size=35&id=XVNvUWCvvlD9&format=png&color=000000"/></a>
                <a href="https://mgjillanimughal.github.io/" target="_blank"><img src="https://img.icons8.com/?size=35&id=AfM2kzPzTz6Q&format=png&color=000000"/></a>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main function to handle the Streamlit application."""
    setup_sidebar()
    # Define navigation options
    choice = st.sidebar.radio("Navigation", [
        "Upload", "Profiling", "Chat With Data", "Preprocessing",
        "Feature Engineering", "Modelling", "Model HyperTuning",
        "Model Explainability", "Model Evaluation", "Unsupervised Learning",
        "Download", "Contact Us", "Export Data"
    ])

    # Map navigation choices to corresponding functions
    nav_functions = {
        "Upload": upload_data,
        "Profiling": perform_profiling,
        "Chat With Data": chat_with_data,
        "Preprocessing": preprocess_data,
        "Feature Engineering": feature_engineering,
        "Modelling": perform_modelling,
        "Model HyperTuning": model_hyper_tuning,
        "Model Explainability": model_explainability,
        "Model Evaluation": model_evaluation,
        "Unsupervised Learning": unsupervised_learning,
        "Download": download_model,
        "Contact Us": contact_us,
        "Export Data": export_data
    }

    # Execute the selected function
    if choice in nav_functions:
        nav_functions[choice]()
    else:
        st.error("Navigation choice not recognized!")


def upload_data():
    """Handle data upload and store the dataset in session state."""
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=['csv', 'xlsx'])
    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            df.to_csv(DATA_FILE, index=False)
            st.session_state['data_uploaded'] = True
            st.session_state['df'] = df
            st.success("File uploaded successfully!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"An error occurred while uploading the file: {e}")


def perform_profiling():
    """Perform exploratory data analysis using pandas-profiling."""
    st.title("Exploratory Data Analysis")
    if st.session_state['data_uploaded']:
        df = st.session_state['df']
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.error("Please upload a dataset first!")


def chat_with_data():
    """Allow users to chat with their data using a language model."""
    st.title("Chat With Your Data")
    if st.session_state['data_uploaded']:
        df = st.session_state['df']
        st.dataframe(df.head())
        query = st.text_area("Enter your query about the data", help="You can ask any analytical question or request visualizations.")
        if st.button("Submit"):
            try:
                result = chat_with_csv(df, query)
                st.success("Query processed successfully.")
                st.write(result)
                chart_info = parse_chart_query(query)
                if chart_info:
                    visualize_data(df, chart_info['chart_type'], chart_info['x_column'], chart_info['y_column'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please upload a dataset first!")


def chat_with_csv(df, query):
    """Interact with the dataset using a language model."""
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ API key is not set. Check your .env file.")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.1)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(query)
    return result


def parse_chart_query(query):
    """Parse the user query to detect chart type and columns."""
    chart_types = {
        "histogram": "Histogram",
        "scatter": "Scatter Plot",
        "box": "Box Plot",
        "line": "Line Plot",
        "bar": "Bar Plot",
        "heatmap": "Heatmap",
        "pair": "Pair Plot"
    }

    chart_type = None
    x_column = None
    y_column = None

    for key in chart_types.keys():
        if key in query.lower():
            chart_type = chart_types[key]
            break

    if chart_type:
        columns = re.findall(r'\b\w+\b', query)
        for col in columns:
            if col in st.session_state['df'].columns:
                if x_column is None:
                    x_column = col
                elif y_column is None:
                    y_column = col
                else:
                    break

    return {'chart_type': chart_type, 'x_column': x_column, 'y_column': y_column} if chart_type else None


def visualize_data(df, chart_type, x_column, y_column):
    """Visualize data based on parsed query information."""
    st.subheader("Data Visualization")
    create_plot(df, x_column, y_column, chart_type)


def create_plot(df, x_column, y_column, plot_type):
    """Create and save a plot based on the specified parameters using Plotly."""
    try:
        if plot_type == "Histogram":
            fig = px.histogram(df, x=x_column, title=f'Histogram of {x_column}')
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df, x=x_column, y=y_column, title=f'Scatter Plot between {x_column} and {y_column}')
        elif plot_type == "Box Plot":
            fig = px.box(df, x=x_column, title=f'Box Plot of {x_column}')
        elif plot_type == "Line Plot":
            fig = px.line(df, x=x_column, y=y_column, title=f'Line Plot between {x_column} and {y_column}')
        elif plot_type == "Bar Plot":
            fig = px.bar(df, x=x_column, y=y_column, title=f'Bar Plot between {x_column} and {y_column}')
        elif plot_type == "Heatmap":
            fig = px.imshow(df.corr(), text_auto=True, title='Heatmap of Correlation Matrix')
        elif plot_type == "Pair Plot":
            fig = px.scatter_matrix(df)

        st.plotly_chart(fig)
        st.success(f"{plot_type} displayed successfully!")
    except Exception as e:
        st.error(f"An error occurred while creating the plot: {e}")


def preprocess_data():
    """Handle data preprocessing tasks such as missing values and encoding."""
    st.title("Data Preprocessing")
    if st.session_state['data_uploaded']:
        df = st.session_state['df']

        # Removing NaN values
        st.subheader("Handling Missing Values")
        st.write("Number of missing values by columns:")
        st.write(df.isnull().sum())
        options = st.multiselect("Choose columns to drop (with too many missing values)", df.columns, default=[])
        df.drop(columns=options, inplace=True)
        st.write("Dropping selected columns...")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.write("Dropped rows with missing values")

        # Handling categorical data
        st.subheader("Handling Categorical Data")
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write("Categorical Columns:", obj_cols)
        encode = st.button("Encode Categorical Data")
        if encode:
            for col in obj_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.write("Categorical columns encoded.")

        # Save processed data
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        st.session_state['processed_data'] = True
        st.session_state['df'] = df
        st.success("Data preprocessing completed!")
        st.dataframe(df)
    else:
        st.error("Please upload a dataset first!")


def feature_engineering():
    """Feature engineering tasks including creation and selection of features."""
    st.title("Feature Engineering")
    if st.session_state['processed_data']:
        df = st.session_state['df']
        st.dataframe(df.head())

        # Creating new features
        st.subheader("Create New Features")
        new_feature_name = st.text_input("New Feature Name")
        new_feature_formula = st.text_input("New Feature Formula", help="Use column names and mathematical operators, e.g., 'col1 + col2'")

        if st.button("Create Feature"):
            try:
                df[new_feature_name] = df.eval(new_feature_formula)
                st.success(f"Feature '{new_feature_name}' created successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"An error occurred: {e}")

        # Feature selection
        st.subheader("Feature Selection")
        selected_features = st.multiselect("Select Features to Keep", df.columns.tolist(), default=df.columns.tolist())

        if st.button("Apply Feature Selection"):
            try:
                df = df[selected_features]
                df.to_csv(PROCESSED_DATA_FILE, index=False)
                st.session_state['df'] = df
                st.success("Feature selection applied!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"An error occurred: {e}")

        # Advanced feature engineering: Polynomial features
        st.subheader("Advanced Feature Engineering: Polynomial Features")
        poly_degree = st.slider("Select Degree for Polynomial Features", 2, 5, 2)
        generate_poly = st.button("Generate Polynomial Features")

        if generate_poly:
            try:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                num_cols = df.select_dtypes(include=['float64', 'int']).columns.tolist()
                df_poly = pd.DataFrame(poly.fit_transform(df[num_cols]), columns=poly.get_feature_names(num_cols))
                df = pd.concat([df, df_poly], axis=1)
                st.success(f"Polynomial features of degree {poly_degree} created successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"An error occurred while creating polynomial features: {e}")
    else:
        st.error("Please preprocess the dataset first!")


def perform_modelling():
    """Handle model training for classification, regression, and time series forecasting."""
    st.title("Model Training")
    if st.session_state['processed_data']:
        df = st.session_state['df']
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression", "Time Series Forecasting"])

        # Choosing train-test split ratio
        split_ratio = st.slider("Select Train-Test Split Ratio", 0.1, 0.9, 0.7)

        if st.button('Run Modelling'):
            try:
                if problem_type == "Classification":
                    run_classification_model(df, chosen_target, split_ratio)
                elif problem_type == "Regression":
                    run_regression_model(df, chosen_target, split_ratio)
                elif problem_type == "Time Series Forecasting":
                    run_time_series_model(df, chosen_target, split_ratio)
            except Exception as e:
                st.error(f"An error occurred during modelling: {e}")
    else:
        st.error("Please preprocess the dataset first!")


def run_classification_model(df, target, split_ratio):
    """Run and save a classification model."""
    exp_clf = ClassificationExperiment()
    exp_clf.setup(data=df, target=target, train_size=split_ratio, session_id=3)
    setup_clf_df = exp_clf.pull()
    st.dataframe(setup_clf_df)

    best_model = exp_clf.compare_models(sort="F1")
    compare_clf_df = exp_clf.pull()
    st.dataframe(compare_clf_df)

    model_file = f"{MODEL_FILE_BASE}_clf.pkl"
    exp_clf.save_model(best_model, model_file)
    st.session_state['model_file'] = model_file
    st.success(f"Classification modelling completed! Best model saved as {model_file}")


def run_regression_model(df, target, split_ratio):
    """Run and save a regression model."""
    exp_reg = RegressionExperiment()
    exp_reg.setup(data=df, target=target, train_size=split_ratio, session_id=3, normalize=True)
    setup_reg_df = exp_reg.pull()
    st.dataframe(setup_reg_df)

    best_model = exp_reg.compare_models(sort="R2")
    compare_reg_df = exp_reg.pull()
    st.dataframe(compare_reg_df)

    model_file = f"{MODEL_FILE_BASE}_reg.pkl"
    exp_reg.save_model(best_model, model_file)
    st.session_state['model_file'] = model_file
    st.success(f"Regression modelling completed! Best model saved as {model_file}")


def run_time_series_model(df, target, split_ratio):
    """Run and save a time series forecasting model."""
    exp_ts = TSForecastingExperiment()
    exp_ts.setup(data=df, target=target, train_size=split_ratio, session_id=3)
    setup_ts_df = exp_ts.pull()
    st.dataframe(setup_ts_df)

    best_model = exp_ts.compare_models()
    compare_ts_df = exp_ts.pull()
    st.dataframe(compare_ts_df)

    model_file = f"{MODEL_FILE_BASE}_ts.pkl"
    exp_ts.save_model(best_model, model_file)
    st.session_state['model_file'] = model_file
    st.success(f"Time Series Forecasting modelling completed! Best model saved as {model_file}")


def model_hyper_tuning():
    """Handle hyperparameter tuning for classification, regression, and time series models."""
    st.title("Model Hyper Tuning")
    if st.session_state['processed_data']:
        df = st.session_state['df']
        chosen_target = st.selectbox("Select Target Variable", df.columns)

        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression", "Time Series Forecasting"])
        split_ratio = st.slider("Select Train-Test Split Ratio", 0.1, 0.9, 0.7)

        if st.button('Run Hyper-Tuning'):
            try:
                if problem_type == "Classification":
                    tune_classification_model(df, chosen_target, split_ratio)
                elif problem_type == "Regression":
                    tune_regression_model(df, chosen_target, split_ratio)
                elif problem_type == "Time Series Forecasting":
                    tune_time_series_model(df, chosen_target, split_ratio)
            except Exception as e:
                st.error(f"An error occurred during hyper-tuning: {e}")
    else:
        st.error("Please preprocess the dataset first!")


def tune_classification_model(df, target, split_ratio):
    """Tune and save a classification model."""
    exp_clf = ClassificationExperiment()
    exp_clf.setup(data=df, target=target, train_size=split_ratio, session_id=3)
    best_model = exp_clf.compare_models()
    tuned_model = exp_clf.tune_model(best_model, optimize="F1", choose_better=True, fold=10)
    scores = exp_clf.pull()

    model_file = f"{MODEL_FILE_BASE}_tuned_clf.pkl"
    exp_clf.save_model(tuned_model, model_file)
    st.session_state['model_file'] = model_file

    st.dataframe(scores)
    st.success(f"Classification hyper-tuning completed! Best hyper-tuned model saved as {model_file}")


def tune_regression_model(df, target, split_ratio):
    """Tune and save a regression model."""
    exp_reg = RegressionExperiment()
    exp_reg.setup(data=df, target=target, train_size=split_ratio, session_id=3, normalize=True)
    best_model = exp_reg.compare_models()
    tuned_model = exp_reg.tune_model(best_model, optimize="R2", choose_better=True, fold=10)
    scores = exp_reg.pull()
    st.dataframe(scores)

    model_file = f"{MODEL_FILE_BASE}_tuned_reg.pkl"
    exp_reg.save_model(tuned_model, model_file)
    st.session_state['model_file'] = model_file

    st.success(f"Regression hyper-tuning completed! Best hyper-tuned model saved as {model_file}")


def tune_time_series_model(df, target, split_ratio):
    """Tune and save a time series forecasting model."""
    exp_ts = TSForecastingExperiment()
    exp_ts.setup(data=df, target=target, train_size=split_ratio, session_id=3)
    best_model = exp_ts.compare_models()
    tuned_model = exp_ts.tune_model(best_model, optimize="MAPE", choose_better=True, fold=10)
    scores = exp_ts.pull()

    model_file = f"{MODEL_FILE_BASE}_tuned_ts.pkl"
    exp_ts.save_model(tuned_model, model_file)
    st.session_state['model_file'] = model_file

    st.dataframe(scores)
    st.success(f"Time Series hyper-tuning completed! Best hyper-tuned model saved as {model_file}")


def model_explainability():
    """Handle model explainability using SHAP values."""
    st.title("Model Explainability")
    if st.session_state['model_file'] and st.session_state['processed_data']:
        df = st.session_state['df']
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        model_type = st.selectbox("Select Model Type", ["Classification", "Regression", "Time Series Forecasting"])

        if st.button('Explain Model'):
            try:
                model_file = st.session_state['model_file']
                if model_type == "Classification":
                    model = ClassificationExperiment().load_model(model_file)
                elif model_type == "Regression":
                    model = RegressionExperiment().load_model(model_file)
                else:
                    model = TSForecastingExperiment().load_model(model_file)

                explain_model(df, chosen_target, model)
            except Exception as e:
                st.error(f"An error occurred during model explainability: {e}")


def explain_model(df, target, model):
    """Explain model predictions using SHAP values."""
    if hasattr(model, 'steps'):
        preprocessing = Pipeline(model.steps[:-1])
        trained_model = model.named_steps['trained_model']
    else:
        preprocessing = None
        trained_model = model

    if preprocessing is not None:
        df_processed = preprocessing.transform(df.drop(columns=[target]))
    else:
        df_processed = df.drop(columns=[target])

    explainer = shap.KernelExplainer(trained_model.predict, df_processed)
    shap_values = explainer.shap_values(df_processed)
    fig = px.bar(x=df_processed.columns, y=shap_values.mean(axis=0), title="Feature Importance")
    st.plotly_chart(fig)


def model_evaluation():
    """Evaluate the performance of the trained model."""
    st.title("Model Evaluation")
    if st.session_state['model_file'] and st.session_state['processed_data']:
        df = st.session_state['df']
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        model_type = st.selectbox("Select Model Type", ["Classification", "Regression", "Time Series Forecasting"])

        if st.button('Evaluate Model'):
            try:
                model_file = st.session_state['model_file']
                if model_type == "Classification":
                    evaluate_classification_model(df, chosen_target, model_file)
                elif model_type == "Regression":
                    evaluate_regression_model(df, chosen_target, model_file)
                elif model_type == "Time Series Forecasting":
                    evaluate_time_series_model(df, chosen_target, model_file)
            except Exception as e:
                st.error(f"An error occurred during model evaluation: {e}")
    else:
        st.error("Please preprocess the dataset and perform modelling first!")


def evaluate_regression_model(df, target, model_file):
    """Evaluate the performance of a regression model."""
    exp = RegressionExperiment()
    try:
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df.dropna(subset=[target], inplace=True)
    except Exception as e:
        st.error(f"Error converting target column to numeric: {e}")
        return

    exp.setup(data=df, target=target, train_size=0.7, session_id=3, normalize=True)
    model = exp.load_model(model_file)

    # Evaluate the model
    exp.evaluate_model(model)

    # Pull evaluation results into a DataFrame
    eval_results = exp.pull()

    if eval_results is not None and isinstance(eval_results, pd.DataFrame):
        st.write("Model Evaluation Results:")
        st.dataframe(eval_results)

        # Extract and display specific metrics
        try:
            r2_score = eval_results.loc[eval_results['Metric'] == 'R2', 'Value'].values[0]
            mae = eval_results.loc[eval_results['Metric'] == 'MAE', 'Value'].values[0]
            mse = eval_results.loc[eval_results['Metric'] == 'MSE', 'Value'].values[0]
            rmse = eval_results.loc[eval_results['Metric'] == 'RMSE', 'Value'].values[0]
            mape = eval_results.loc[eval_results['Metric'] == 'MAPE', 'Value'].values[0]

            st.write(f"R² Score: {r2_score:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

        except KeyError as e:
            st.error(f"Expected evaluation metric not found in results: {e}")
    else:
        st.error("Evaluation results are not available or not in expected format.")

    # Plot residuals for visual inspection of model performance
    plot_residuals(exp, model, df, target)


def plot_residuals(exp, model, df, target):
    """Plot residuals for a regression model."""
    try:
        # Getting predictions
        y_pred_df = exp.predict_model(model, data=df)

        # Extract predictions from the DataFrame
        if 'Label' in y_pred_df.columns:
            y_pred = y_pred_df['Label']
        elif 'Prediction' in y_pred_df.columns:
            y_pred = y_pred_df['Prediction']
        else:
            y_pred = y_pred_df.iloc[:, -1]  # Fallback to the last column for predictions

        y_true = df[target].values

        # Convert predictions and true values to NumPy arrays
        y_pred = y_pred.to_numpy()
        y_true = y_true.flatten()  # NumPy arrays have .flatten()

        residuals = y_true - y_pred

        # Plot residuals
        fig = px.scatter(x=y_true, y=residuals, labels={'x': 'True Values', 'y': 'Residuals'}, title='Residuals Plot')
        st.plotly_chart(fig)

        # Display residual statistics
        st.write(f"Mean Residual: {residuals.mean():.4f}")
        st.write(f"Residual Variance: {residuals.var():.4f}")

    except Exception as e:
        st.error(f"An error occurred while plotting residuals: {e}")


def evaluate_classification_model(df, target, model_file):
    """Evaluate the performance of a classification model."""
    exp = ClassificationExperiment()
    exp.setup(data=df, target=target, train_size=0.7, session_id=3)
    model = exp.load_model(model_file)

    # Evaluate model and display results
    eval_results = exp.evaluate_model(model)
    st.write(eval_results)

    # Display accuracy
    accuracy = exp.pull()["Accuracy"]
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    plot_confusion_matrix(exp, model)


def plot_confusion_matrix(exp, model):
    """Plot the confusion matrix for a classification model."""
    try:
        fig = exp.plot_model(model, plot='confusion_matrix')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while plotting the confusion matrix: {e}")


def evaluate_time_series_model(df, target, model_file):
    """Evaluate the performance of a time series forecasting model."""
    exp = TSForecastingExperiment()
    exp.setup(data=df, target=target, train_size=0.7, session_id=3)
    model = exp.load_model(model_file)

    # Evaluate model and display results
    eval_results = exp.evaluate_model(model)
    st.write(eval_results)

    # Plot forecast results
    plot_forecast(exp, model, df, target)


def plot_forecast(exp, model, df, target):
    """Plot the forecast results for a time series model."""
    try:
        forecast_df = exp.predict_model(model, data=df)
        st.write("Forecast DataFrame:", forecast_df.head())

        # Plot true values vs. forecasted values
        fig = px.line(df, x=df.index, y=[target], title='Actual vs. Forecasted Values')
        fig.add_scatter(x=df.index, y=forecast_df['Label'], mode='lines', name='Forecast')
        st.plotly_chart(fig)

        # Optionally, display MAPE, RMSE, etc.
        st.write(f"MAPE: {forecast_df['MAPE'].mean():.4f}")

    except Exception as e:
        st.error(f"An error occurred while plotting the forecast: {e}")


def unsupervised_learning():
    """Handle unsupervised learning tasks such as clustering and PCA."""
    st.title("Unsupervised Learning")
    if st.session_state['processed_data']:
        df = st.session_state['df']
        st.dataframe(df.head())

        unsupervised_option = st.selectbox("Select Unsupervised Learning Technique", ["Clustering", "PCA"])

        if unsupervised_option == "Clustering":
            perform_clustering(df)
        elif unsupervised_option == "PCA":
            perform_pca(df)
    else:
        st.error("Please preprocess the dataset first!")


def perform_clustering(df):
    """Perform clustering using K-Means."""
    st.subheader("Clustering")
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
    selected_features = st.multiselect("Select Features for Clustering", df.columns, default=df.columns.tolist())

    if st.button("Run Clustering"):
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[selected_features])
            st.success(f"Clustering completed! Number of clusters: {num_clusters}")
            st.dataframe(df)

            plot_cluster_results(df, selected_features, 'Cluster')
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")


def perform_pca(df):
    """Perform Principal Component Analysis (PCA)."""
    st.subheader("Principal Component Analysis (PCA)")
    selected_features = st.multiselect("Select Features for PCA", df.columns, default=df.columns.tolist())
    n_components = st.slider("Number of Principal Components", 2, len(selected_features), 2)

    if st.button("Run PCA"):
        try:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[selected_features])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df_scaled)

            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            st.success(f"PCA completed with {n_components} components!")
            st.dataframe(pca_df)

            plot_pca_results(pca_df)
        except Exception as e:
            st.error(f"An error occurred during PCA: {e}")


def plot_cluster_results(df, features, cluster_column):
    """Visualize the clustering results using scatter plot."""
    if len(features) >= 2:
        fig = px.scatter(df, x=features[0], y=features[1], color=cluster_column, title="Clustering Results")
        st.plotly_chart(fig)
    else:
        st.error("Please select at least 2 features for visualization.")


def plot_pca_results(pca_df):
    """Visualize the PCA results using scatter plot."""
    if pca_df.shape[1] >= 2:
        fig = px.scatter(pca_df, x='PC1', y='PC2', title="PCA Results")
        st.plotly_chart(fig)
    else:
        st.error("PCA results require at least 2 components for visualization.")


def download_model():
    """Allow users to download the trained model."""
    st.title("Download Model")
    if st.session_state['model_file']:
        model_file = st.session_state['model_file']
        with open(model_file, 'rb') as f:
            bytes_data = f.read()
            st.download_button('Download Model', bytes_data, file_name=os.path.basename(model_file))
    else:
        st.error("Please perform modelling first to generate a model!")


def contact_us():
    """Display contact information."""
    st.title("Contact Us")
    st.write("For any inquiries regarding our website or services, please feel free to contact us through the following channels:")
    st.markdown("""
        - **Email:** m.g.jillani123@gmail.com
        - **Phone:** +92-321-1174167
        - **Address:** Gulberg III, Lahore, Pakistan
        - **LinkedIn:** [Jillani Soft Tech](https://www.linkedin.com/in/jillanisofttech/)
    """)
    st.write("We look forward to hearing from you!")


def export_data():
    """Allow users to export the cleaned and processed dataset."""
    st.title("Export Data")
    if st.session_state['processed_data']:
        df = st.session_state['df']
        with open(EXPORT_DATA_FILE, 'w') as f:
            df.to_csv(f, index=False)
        with open(EXPORT_DATA_FILE, 'rb') as f:
            st.download_button("Download Processed Data", f, file_name=EXPORT_DATA_FILE)
    else:
        st.error("Please preprocess the data first!")


# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
