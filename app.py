import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import io
import base64

# Set page configuration
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Functions for seaborn dataset loading
def load_seaborn_dataset(dataset_name):
    try:
        data = sns.load_dataset(dataset_name)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# List of available seaborn datasets
seaborn_datasets = [
    "penguins", "tips", "iris", "diamonds", "titanic",
    "mpg", "planets", "car_crashes", "attention", "flights"
]

# Function to determine if a problem is classification or regression
def determine_problem_type(target_variable, data):
    unique_values = data[target_variable].nunique()
    if unique_values <= 10 and pd.api.types.is_categorical_dtype(data[target_variable]) or \
       pd.api.types.is_object_dtype(data[target_variable]) or \
       unique_values <= 5:
        return "Classification"
    else:
        return "Regression"

# Download model function
def download_model():
    import pickle
    output = pickle.dumps(st.session_state.model)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download Model</a>'
    return href

# Main App
st.title("ML Model Trainer Application")

# Sidebar for dataset selection and feature selection
with st.sidebar:
    st.header("Configuration")
    
    # Dataset selection
    st.subheader("Step 1: Select Dataset")
    dataset_option = st.radio(
        "Choose dataset source:",
        ["Seaborn Datasets", "Upload CSV"]
    )
    
    if dataset_option == "Seaborn Datasets":
        selected_dataset = st.selectbox("Select Dataset", seaborn_datasets)
        if st.button("Load Dataset"):
            data = load_seaborn_dataset(selected_dataset)
            if data is not None:
                st.session_state.data = data
                st.session_state.dataset_name = selected_dataset
                st.success(f"Dataset '{selected_dataset}' loaded successfully!")
                # Reset model training state
                st.session_state.model_trained = False
                st.session_state.model = None
                st.session_state.predictions = None
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.dataset_name = uploaded_file.name
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
                # Reset model training state
                st.session_state.model_trained = False
                st.session_state.model = None
                st.session_state.predictions = None
            except Exception as e:
                st.error(f"Error: {e}")

# Main content
if st.session_state.data is not None:
    st.header(f"Dataset: {st.session_state.dataset_name}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.data.head())
    
    # Dataset information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Info")
        buffer = io.StringIO()
        st.session_state.data.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        st.subheader("Dataset Statistics")
        st.write(st.session_state.data.describe())

    # Feature selection
    st.header("Feature Selection")
    
    # Separate numerical and categorical features
    numerical_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Target Variable")
        target_variable = st.selectbox("Target", numerical_cols + categorical_cols)
    
    with col2:
        problem_type = "Unknown"
        if target_variable:
            problem_type = determine_problem_type(target_variable, st.session_state.data)
            st.subheader("Problem Type")
            st.write(f"**{problem_type}**")
    
    # Feature selection widgets
    st.subheader("Select Features")
    
    available_features = [col for col in st.session_state.data.columns if col != target_variable]
    
    with st.form("feature_selection_form"):
        st.subheader("Quantitative Features")
        selected_numerical_features = st.multiselect(
            "Select numerical features",
            [col for col in numerical_cols if col != target_variable]
        )
        
        st.subheader("Qualitative Features")
        selected_categorical_features = st.multiselect(
            "Select categorical features",
            [col for col in categorical_cols if col != target_variable]
        )
        
        # Only proceed if features are selected
        selected_features = selected_numerical_features + selected_categorical_features
        
        # Model selection
        st.subheader("Select Model")
        model_type = st.selectbox(
            "Choose model type",
            ["Linear Regression", "Random Forest"] if problem_type == "Regression" else 
            ["Random Forest"]
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        
        # Specific model parameters
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 30, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        elif model_type == "Linear Regression":
            fit_intercept = st.checkbox("Fit Intercept", True)
            normalize = st.checkbox("Normalize", False)
        
        # Submit button (Fit)
        fit_pressed = st.form_submit_button("Fit Model")
    
    # Process after form submission
    if fit_pressed and selected_features and target_variable:
        with st.spinner('Training model...'):
            # Prepare the data
            X = st.session_state.data[selected_features].copy()
            y = st.session_state.data[target_variable].copy()
            
            # Handle categorical features
            X_processed = pd.get_dummies(X, drop_first=True)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
            
            # Train the model based on selection
            if problem_type == "Regression":
                if model_type == "Linear Regression":
                    model = LinearRegression(fit_intercept=fit_intercept)
                    if normalize:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                else:  # Random Forest
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
            else:  # Classification
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.predictions = y_pred
            st.session_state.model_trained = True
            st.session_state.model_type = model_type
            st.session_state.problem_type = problem_type
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            # Get feature importance if applicable
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X_processed.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.session_state.feature_importance = feature_importance
            
            st.success('Model training completed!')

    # Display results after training
    if st.session_state.model_trained:
        st.header("Model Performance")
        
        # Metrics
        if st.session_state.problem_type == "Regression":
            # Regression metrics
            mse = mean_squared_error(st.session_state.y_test, st.session_state.predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
                
            # Residual plot
            st.subheader("Residual Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            residuals = st.session_state.y_test - st.session_state.predictions
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Residual Distribution")
            ax.set_xlabel("Residual Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
        else:
            # Classification metrics
            accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
            cm = confusion_matrix(st.session_state.y_test, st.session_state.predictions)
            
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(st.session_state.y_test, st.session_state.predictions, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            # ROC Curve (only for binary classification)
            if len(np.unique(st.session_state.y_test)) == 2 and hasattr(st.session_state.model, 'predict_proba'):
                st.subheader("ROC Curve")
                from sklearn.metrics import roc_curve, auc
                y_prob = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
                st.pyplot(fig)
        
        # Feature Importance
        if st.session_state.feature_importance is not None:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = st.session_state.feature_importance.head(10)  # Show top 10 features
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
            ax.set_title("Top Feature Importance")
            st.pyplot(fig)

        # Model Download Option (Advanced Feature)
        st.subheader("Model Export")
        st.markdown(download_model(), unsafe_allow_html=True)

else:
    # Initial app state - no data loaded
    st.info("Please select a dataset from the sidebar to begin.")
    
    # Display instructions
    st.header("Instructions")
    st.write("""
    1. **Select a Dataset**: Choose from available Seaborn datasets or upload your own CSV file.
    2. **Select Features**: Choose which features to use for model training.
    3. **Configure Model**: Choose model type and set parameters.
    4. **Train Model**: Click 'Fit Model' to train.
    5. **Evaluate Results**: View performance metrics and visualizations.
    """)
    
    # Show example images
    st.subheader("Example Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.random.normal(0, 1, 100)
        y = x * 3 + np.random.normal(0, 1, 100)
        ax.scatter(x, y)
        ax.set_title("Example Scatter Plot")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = np.random.randn(100)
        ax.hist(data, bins=20)
        ax.set_title("Example Histogram")
        st.pyplot(fig)