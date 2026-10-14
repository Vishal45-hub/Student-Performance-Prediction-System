
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<div class="main-header">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Predict student performance based on various academic and personal factors</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Prediction", "Model Info", "About"])

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects."""
    try:
        model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_gender = joblib.load('label_encoder_gender.pkl')
        le_stream = joblib.load('label_encoder_stream.pkl')
        return model, scaler, le_gender, le_stream
    except FileNotFoundError:
        st.error("Model files not found. Please run the main script first to train the model.")
        return None, None, None, None

# Load data for statistics
@st.cache_data
def load_data():
    """Load the processed dataset for displaying statistics."""
    try:
        df = pd.read_csv('student_performance_processed.csv')
        return df
    except FileNotFoundError:
        return None

# Home Page
if page == "Home":
    st.markdown('<div class="sub-header">Welcome to the Student Performance Prediction System</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("""
        ### 🎯 **Objective**
        This system predicts student performance (final marks) based on various factors including:

        - **Study Hours**: Daily hours spent studying
        - **Attendance**: Percentage of classes attended
        - **Sleep Duration**: Average daily sleep hours
        - **Previous Scores**: Scores from previous assessments
        - **Assignment Scores**: Marks obtained in assignments
        - **Extracurricular Activities**: Participation in non-academic activities
        - **Parental Education**: Education level of parents
        - **Distance from School**: Distance between home and school

        ### 📊 **How it Works**
        1. **Data Collection**: Gather student information
        2. **Preprocessing**: Clean and prepare the data
        3. **Model Training**: Use Linear Regression to learn patterns
        4. **Prediction**: Predict final marks for new students
        """)

    with col2:
        # Display some statistics if data is available
        df = load_data()
        if df is not None:
            st.write("### 📈 **Dataset Statistics**")

            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.metric("Total Students", f"{len(df):,}")
                st.metric("Average Marks", f"{df['Final_Marks'].mean():.1f}")
                st.metric("Highest Marks", f"{df['Final_Marks'].max():.1f}")

            with col_stats2:
                st.metric("Gender Distribution", f"{df['Gender'].value_counts().iloc[0]} / {df['Gender'].value_counts().iloc[1]}")
                st.metric("Average Study Hours", f"{df['Study_Hours'].mean():.1f} hrs")
                st.metric("Average Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")

        # Display project files
        st.write("### 📁 **Project Files**")
        files = [
            "student_performance_raw.csv",
            "student_performance_processed.csv",
            "linear_regression_model.pkl",
            "scaler.pkl",
            "app.py",
            "student_performance_prediction.py"
        ]

        for file in files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")

# Prediction Page
elif page == "Prediction":
    st.markdown('<div class="sub-header">🔮 Make a Prediction</div>', unsafe_allow_html=True)

    # Load model
    model, scaler, le_gender, le_stream = load_model()

    if model is None:
        st.stop()

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📝 Student Information")

        gender = st.selectbox("Gender", ["Male", "Female"])
        stream = st.selectbox("Stream", ["Science", "Commerce", "Arts"])
        parental_education = st.selectbox("Parental Education Level",
                                        ["High School", "Bachelor's", "Master's", "PhD"],
                                        help="Education level of the student's parents")

        st.write("---")

        study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 4.0,
                              help="Average hours spent studying daily")
        attendance = st.slider("Attendance Percentage", 40.0, 100.0, 85.0,
                             help="Percentage of classes attended")
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.5,
                                 help="Average hours of sleep per day")

    with col2:
        st.write("### 📊 Academic Performance")

        previous_scores = st.number_input("Previous Scores", 30.0, 95.0, 75.0,
                                        help="Scores from previous assessments")
        assignment_scores = st.number_input("Assignment Scores", 20.0, 100.0, 80.0,
                                          help="Marks obtained in assignments")
        extracurricular = st.slider("Extracurricular Activities", 0, 5, 2,
                                  help="Number of extracurricular activities participated in")
        distance_from_school = st.number_input("Distance from School (km)", 0.5, 15.0, 5.0,
                                             help="Distance between home and school in kilometers")

    # Predict button
    if st.button("🔮 Predict Final Marks", type="primary"):
        # Prepare the input data
        input_data = np.array([[
            study_hours,
            attendance,
            sleep_duration,
            previous_scores,
            assignment_scores,
            extracurricular,
            {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}[parental_education],
            distance_from_school,
            le_gender.transform([gender])[0],
            le_stream.transform([stream])[0]
        ]])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Display prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="color: #1f77b4;">Predicted Final Marks: {prediction:.2f}/100</h2>
        </div>
        """, unsafe_allow_html=True)

        # Add interpretation
        if prediction >= 90:
            st.success("Excellent performance expected! 🌟")
        elif prediction >= 80:
            st.info("Very good performance expected! 👍")
        elif prediction >= 70:
            st.warning("Good performance expected. Keep improving! 💪")
        elif prediction >= 60:
            st.warning("Average performance. Needs attention! ⚠️")
        else:
            st.error("Below average performance. Immediate intervention needed! 🚨")

        st.markdown('</div>', unsafe_allow_html=True)

        # Show feature importance for this prediction
        st.write("### 🔍 **Factors Contributing to this Prediction**")

        # Get feature coefficients
        feature_names = ['Study_Hours', 'Attendance_Percentage', 'Sleep_Duration',
                        'Previous_Scores', 'Assignment_Scores', 'Extracurricular_Activities',
                        'Parental_Education', 'Distance_From_School', 'Gender_Encoded', 'Stream_Encoded']

        # Create a DataFrame for feature contributions
        contributions = []
        for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
            # Get the scaled value for this feature
            scaled_value = input_scaled[0][i]
            contribution = coef * scaled_value
            contributions.append({
                'Feature': name,
                'Contribution': contribution,
                'Coefficient': coef,
                'Value': input_data[0][i]
            })

        contrib_df = pd.DataFrame(contributions).sort_values('Contribution', key=abs, ascending=False)

        # Display top contributing factors
        col_contrib1, col_contrib2 = st.columns(2)

        with col_contrib1:
            st.write("**Top Positive Contributors:**")
            pos_contrib = contrib_df[contrib_df['Contribution'] > 0].head(3)
            for _, row in pos_contrib.iterrows():
                st.write(f"- {row['Feature']}: +{row['Contribution']:.2f}")

        with col_contrib2:
            st.write("**Top Negative Contributors:**")
            neg_contrib = contrib_df[contrib_df['Contribution'] < 0].head(3)
            for _, row in neg_contrib.iterrows():
                st.write(f"- {row['Feature']}: {row['Contribution']:.2f}")

# Model Info Page
elif page == "Model Info":
    st.markdown('<div class="sub-header">📊 Model Information</div>', unsafe_allow_html=True)

    # Load data for model evaluation
    df = load_data()
    if df is not None:
        st.write("### 📈 **Model Performance Metrics**")

        # Load model for evaluation
        model, scaler, le_gender, le_stream = load_model()

        if model is not None:
            # Prepare data for evaluation
            features = ['Study_Hours', 'Attendance_Percentage', 'Sleep_Duration',
                       'Previous_Scores', 'Assignment_Scores', 'Extracurricular_Activities',
                       'Parental_Education', 'Distance_From_School', 'Gender_Encoded', 'Stream_Encoded']

            X = df[features].values
            y = df['Final_Marks'].values

            # Scale features
            X_scaled = scaler.transform(X)

            # Make predictions
            y_pred = model.predict(X_scaled)

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Mean Absolute Error</h4>
                    <h2 style="color: #1f77b4;">{mae:.2f}</h2>
                    <p style="color: #666;">Average prediction error</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Root Mean Squared Error</h4>
                    <h2 style="color: #1f77b4;">{rmse:.2f}</h2>
                    <p style="color: #666;">Standard deviation of errors</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>R² Score</h4>
                    <h2 style="color: #1f77b4;">{r2:.4f}</h2>
                    <p style="color: #666;">Variance explained</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Model Type</h4>
                    <h2 style="color: #1f77b4;">Linear Regression</h2>
                    <p style="color: #666;">Machine Learning Algorithm</p>
                </div>
                """, unsafe_allow_html=True)

            # Feature importance
            st.write("### 🔍 **Feature Importance**")

            feature_importance = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)

            # Create visualization
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff9999' if x < 0 else '#66b3ff' for x in feature_importance['Coefficient']]
            ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Feature Importance (Linear Regression Coefficients)')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()

            st.pyplot(fig)

            # Display coefficient table
            st.write("### 📋 **Coefficient Values**")
            st.dataframe(feature_importance, use_container_width=True)

    # Model explanation
    st.write("### 🤖 **How the Model Works**")
    st.write("""
    **Linear Regression** is a statistical method that models the relationship between a dependent variable
    (Final Marks) and one or more independent variables (study hours, attendance, etc.).

    **The Model Equation:**
    ```
    Final_Marks = β₀ + β₁(Study_Hours) + β₂(Attendance) + β₃(Sleep_Duration) + ... + β₁₀(Stream_Encoded)
    ```

    Where:
    - β₀ is the intercept (baseline marks)
    - β₁, β₂, ..., β₁₀ are coefficients representing the impact of each feature

    **Key Assumptions:**
    - Linear relationship between features and target
    - Features are independent of each other
    - Residuals are normally distributed
    """)

# About Page
elif page == "About":
    st.markdown('<div class="sub-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        ### 🎓 **Student Performance Prediction System**

        This is a comprehensive Data Science mini project that demonstrates the complete machine learning pipeline:

        #### 📋 **Project Components**
        1. **Data Generation**: Synthetic dataset creation with realistic student performance factors
        2. **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling
        3. **Exploratory Data Analysis (EDA)**: Statistical analysis and data visualization
        4. **Model Building**: Linear Regression implementation
        5. **Model Evaluation**: Performance metrics and validation
        6. **Web Application**: Interactive Streamlit interface for predictions

        #### 🛠 **Technologies Used**
        - **Python Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
        - **Machine Learning**: Linear Regression
        - **Web Framework**: Streamlit
        - **Data Visualization**: matplotlib, seaborn

        #### 📊 **Dataset Information**
        - **Size**: 500 student records
        - **Features**: 10 input variables
        - **Target**: Final marks (0-100)
        - **Data Quality**: Includes missing values and duplicates for preprocessing demonstration

        #### 🎯 **Learning Objectives**
        This project demonstrates:
        - Complete machine learning workflow
        - Data preprocessing techniques
        - Feature engineering and selection
        - Model evaluation and interpretation
        - Web application development for ML models
        """)

    with col2:
        st.write("### 📈 **Project Statistics**")

        # Load data if available
        df = load_data()
        if df is not None:
            st.metric("Dataset Size", f"{len(df):,} records")
            st.metric("Features", "10 input variables")
            st.metric("Target Range", "0-100 marks")
            st.metric("Gender Distribution", f"{df['Gender'].value_counts().iloc[0]} Male / {df['Gender'].value_counts().iloc[1]} Female")
            st.metric("Stream Distribution", f"{df['Stream'].value_counts().iloc[0]} Science / {df['Stream'].value_counts().iloc[1]} Commerce / {df['Stream'].value_counts().iloc[2]} Arts")

        st.write("### 📁 **File Structure**")
        st.code("""
student_performance_prediction.py  # Main ML script
app.py                            # Streamlit web app
requirements.txt                  # Dependencies
student_performance_raw.csv       # Raw dataset
student_performance_processed.csv # Processed dataset
linear_regression_model.pkl       # Trained model
scaler.pkl                        # Feature scaler
label_encoder_*.pkl              # Label encoders
plots/                           # Visualization directory
        """)

    # Contact and feedback
    st.write("### 📞 **Contact & Feedback**")
    st.write("""
    This project was created as a comprehensive demonstration of machine learning concepts.

    **For questions or feedback:**
    - Review the code comments for detailed explanations
    - Check the main script for complete implementation details
    - Explore the plots directory for generated visualizations

    **Educational Use:**
    This project is suitable for:
    - College-level data science courses
    - Machine learning workshops
    - Self-study and portfolio projects
    """)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem;">© 2024 Student Performance Prediction System | Created for Educational Purposes</div>', unsafe_allow_html=True)