# ML Model Trainer Streamlit Application

This interactive Streamlit application allows users to train and evaluate machine learning models on various datasets without writing code.

## Features

### Basic Features
- Select from a variety of Seaborn datasets or upload custom CSV files
- Choose between quantitative and qualitative features for model training
- Implement multiple model types (Linear Regression, Random Forest)
- Configure model parameters (test size, model-specific parameters)
- View model performance metrics based on model type
- Visualize results with appropriate plots (residual distributions, confusion matrices, etc.)

### Advanced Features
- Custom dataset upload functionality
- Model export/download capability
- User-friendly interface with forms for better experience
- Feature importance visualization

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ml-model-trainer.git
cd ml-model-trainer
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

## Deployment

This application can be deployed to Streamlit Community Cloud by following these steps:

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Select your repository and branch
5. Click "Deploy"

## Usage Instructions

1. **Select a Dataset**: Choose from available Seaborn datasets or upload your own CSV file using the sidebar.
2. **Select Features**: Choose target variable and features for your model.
3. **Configure Model**: Select model type and adjust parameters as needed.
4. **Train Model**: Click the "Fit Model" button to train your model.
5. **Review Results**: Explore performance metrics and visualizations.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

- `app.py`: Main application code
- `requirements.txt`: Required Python packages
- `README.md`: Documentation