import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(train_path):
    # Load the dataset
    train_df = pd.read_csv('train.csv')

    # Drop columns with a high number of missing values
    columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    train_df_dropped = train_df.drop(columns=columns_to_drop)

    # Separate features and target variable
    X = train_df_dropped.drop(columns=['SalePrice'])
    y = train_df_dropped['SalePrice']

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines for numerical and categorical data
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor, numerical_cols, categorical_cols

# Function to load CSS
def load_css(css_file_path):
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main Streamlit app
def main():
    # Load CSS
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    load_css(css_path)
    
    st.title('House Price Prediction')

    # File uploader
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.header('Upload Training Data')
    train_file = st.file_uploader('Upload Training Data (CSV)', type=['csv'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if train_file:
        X, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess_data(train_file)

        # Split data into train and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

        # Define the model
        model = RandomForestRegressor(n_estimators=100, random_state=0)

        # Create and evaluate the pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        # Preprocessing of training data, fit model
        pipeline.fit(X_train, y_train)

        # Preprocessing of validation data, get predictions
        preds = pipeline.predict(X_valid)

        # Evaluate the model
        rmse = mean_squared_error(y_valid, preds, squared=False)
        st.write(f'**Model Evaluation**')
        st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

        # Prediction form
        st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
        st.header('Make a Prediction')
        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(f'{col}')

        if st.button('Predict'):
            input_df = pd.DataFrame([input_data])
            input_df[numerical_cols] = input_df[numerical_cols].apply(pd.to_numeric, errors='coerce')
            input_df = preprocessor.transform(input_df)
            prediction = pipeline.predict(input_df)
            st.write(f'**Predicted Sale Price:** ${prediction[0]:,.2f}')
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
