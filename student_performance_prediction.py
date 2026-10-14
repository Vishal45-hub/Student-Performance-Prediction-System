import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("=" * 80)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 80)

# =============================================================================
# 1. PROBLEM STATEMENT
# =============================================================================

print("\n" + "=" * 80)
print("1. PROBLEM STATEMENT")
print("=" * 80)

problem_statement = """
The objective of this project is to predict student performance (final marks)
based on various factors that influence academic success.

Key Factors Considered:
- Study Hours: Daily hours spent studying
- Attendance: Percentage of classes attended
- Sleep Duration: Average daily sleep hours
- Previous Scores: Scores from previous assessments
- Assignment Scores: Marks obtained in assignments
- Extracurricular Activities: Participation in non-academic activities
- Parental Education: Education level of parents
- Distance from School: Distance between home and school (in km)

Target Variable:
- Final Marks: End-of-term examination marks (0-100)

This prediction system can help educators identify at-risk students early and
provide timely interventions to improve academic outcomes.
"""
print(problem_statement)

# =============================================================================
# 2. DATASET GENERATION
# =============================================================================

print("\n" + "=" * 80)
print("2. DATASET GENERATION")
print("=" * 80)

def generate_student_dataset(n_samples=500, random_state=42):
    """
    Generate a synthetic dataset for student performance prediction.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 500)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)

    # Generate base features
    # Study hours per day (1-10 hours)
    study_hours = np.random.uniform(1, 10, n_samples)

    # Attendance percentage (40-100%)
    attendance = np.random.uniform(40, 100, n_samples)

    # Sleep duration (4-10 hours)
    sleep_duration = np.random.uniform(4, 10, n_samples)

    # Previous scores (30-95)
    previous_scores = np.random.uniform(30, 95, n_samples)

    # Assignment scores (20-100)
    assignment_scores = np.random.uniform(20, 100, n_samples)

    # Extracurricular activities (0-5 activities)
    extracurricular = np.random.randint(0, 6, n_samples)

    # Parental education level (1=High School, 2=Bachelor's, 3=Master's, 4=PhD)
    parental_education = np.random.choice([1, 2, 3, 4], n_samples,
                                          p=[0.2, 0.4, 0.3, 0.1])

    # Distance from school (0.5-15 km)
    distance_from_school = np.random.uniform(0.5, 15, n_samples)

    # Gender (Male/Female)
    gender = np.random.choice(['Male', 'Female'], n_samples)

    # Stream (Science, Commerce, Arts)
    stream = np.random.choice(['Science', 'Commerce', 'Arts'], n_samples,
                              p=[0.4, 0.35, 0.25])

    # Calculate final marks with realistic relationships
    # Base marks from previous performance
    base_marks = (previous_scores * 0.3 + assignment_scores * 0.25)

    # Add contribution from study hours (positive correlation)
    study_contribution = study_hours * 2.5

    # Add contribution from attendance (positive correlation)
    attendance_contribution = attendance * 0.15

    # Add contribution from sleep (optimal around 7-8 hours)
    optimal_sleep = 7.5
    sleep_contribution = 5 - abs(sleep_duration - optimal_sleep) * 1.5

    # Parental education contribution
    education_contribution = parental_education * 2

    # Distance contribution (negative - longer distance = slightly lower marks)
    distance_contribution = -distance_from_school * 0.3

    # Extracurricular contribution (moderate positive effect)
    extracurricular_contribution = extracurricular * 0.8

    # Stream adjustment
    stream_adjustment = {'Science': 2, 'Commerce': 0, 'Arts': -1}
    stream_contrib = np.array([stream_adjustment[s] for s in stream])

    # Combine all factors
    final_marks = (base_marks + study_contribution + attendance_contribution +
                   sleep_contribution + education_contribution +
                   distance_contribution + extracurricular_contribution +
                   stream_contrib)

    # Add some random noise
    noise = np.random.normal(0, 3, n_samples)
    final_marks = final_marks + noise

    # Clip marks to valid range (0-100)
    final_marks = np.clip(final_marks, 0, 100)

    # Create DataFrame
    data = pd.DataFrame({
        'Gender': gender,
        'Study_Hours': np.round(study_hours, 1),
        'Attendance_Percentage': np.round(attendance, 1),
        'Sleep_Duration': np.round(sleep_duration, 1),
        'Previous_Scores': np.round(previous_scores, 1),
        'Assignment_Scores': np.round(assignment_scores, 1),
        'Extracurricular_Activities': extracurricular,
        'Parental_Education': parental_education,
        'Distance_From_School': np.round(distance_from_school, 1),
        'Stream': stream,
        'Final_Marks': np.round(final_marks, 1)
    })

    return data

# Generate the dataset
print("\nGenerating synthetic dataset with 500 student records...")
df = generate_student_dataset(n_samples=500, random_state=42)

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn Names:")
for col in df.columns:
    print(f"  - {col}")

# Introduce some missing values and duplicates for preprocessing demonstration
print("\nIntroducing some missing values and duplicates for preprocessing demo...")

# Add missing values (about 5% of data)
missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
for idx in missing_indices[:10]:
    col = np.random.choice(['Study_Hours', 'Attendance_Percentage', 'Sleep_Duration',
                           'Previous_Scores', 'Assignment_Scores'])
    df.loc[idx, col] = np.nan

# Add a few duplicate rows
duplicate_indices = np.random.choice(df.index, size=5, replace=False)
duplicates = df.loc[duplicate_indices]
df = pd.concat([df, duplicates], ignore_index=True)

print(f"\nDataset Shape after adding missing values and duplicates: {df.shape}")

# Save the raw dataset
df.to_csv('student_performance_raw.csv', index=False)
print("\nRaw dataset saved as 'student_performance_raw.csv'")

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("3. DATA PREPROCESSING")
print("=" * 80)

# 3.1 Display initial data
print("\n--- 3.1 Initial Data Overview ---")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe().round(2))

# 3.2 Handle Missing Values
print("\n--- 3.2 Handling Missing Values ---")

print("\nMissing values before handling:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

# Fill missing values with median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {col} missing values with median: {median_val:.2f}")

print("\nMissing values after handling:")
print(f"  Total missing values: {df.isnull().sum().sum()}")

# 3.3 Remove Duplicates
print("\n--- 3.3 Removing Duplicates ---")

duplicates_before = df.duplicated().sum()
print(f"Duplicates found: {duplicates_before}")

df = df.drop_duplicates()
print(f"Duplicates removed. New shape: {df.shape}")

# 3.4 Convert Categorical Data to Numerical
print("\n--- 3.4 Converting Categorical Data to Numerical ---")

# Create a copy for preprocessing
df_processed = df.copy()

# Label Encoding for Gender
le_gender = LabelEncoder()
df_processed['Gender_Encoded'] = le_gender.fit_transform(df_processed['Gender'])
print(f"\nGender Encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# Label Encoding for Stream
le_stream = LabelEncoder()
df_processed['Stream_Encoded'] = le_stream.fit_transform(df_processed['Stream'])
print(f"Stream Encoding: {dict(zip(le_stream.classes_, le_stream.transform(le_stream.classes_)))}")

# Map Parental Education to descriptive labels
education_map = {1: 'High_School', 2: 'Bachelors', 3: 'Masters', 4: 'PhD'}
df_processed['Parental_Education_Label'] = df_processed['Parental_Education'].map(education_map)
print(f"\nParental Education Mapping: {education_map}")

# 3.5 Feature Selection
print("\n--- 3.5 Feature Selection ---")

# Select features for modeling
features = ['Study_Hours', 'Attendance_Percentage', 'Sleep_Duration',
            'Previous_Scores', 'Assignment_Scores', 'Extracurricular_Activities',
            'Parental_Education', 'Distance_From_School', 'Gender_Encoded',
            'Stream_Encoded']

target = 'Final_Marks'

print(f"\nFeatures selected for modeling ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"  {i}. {feat}")
print(f"\nTarget variable: {target}")

# Create feature matrix and target vector
X = df_processed[features].values
y = df_processed[target].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 3.6 Feature Scaling (Normalization)
print("\n--- 3.6 Feature Scaling ---")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling applied using StandardScaler")
print(f"Scaled feature matrix shape: {X_scaled.shape}")

# Save processed data
df_processed.to_csv('student_performance_processed.csv', index=False)
print("\nProcessed dataset saved as 'student_performance_processed.csv'")

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'label_encoder_gender.pkl')
joblib.dump(le_stream, 'label_encoder_stream.pkl')
print("Scaler and Label Encoders saved.")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 80)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Create a directory for saving plots
os.makedirs('plots', exist_ok=True)

# 4.1 Distribution of Final Marks
print("\n--- 4.1 Distribution of Final Marks ---")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_processed['Final_Marks'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Final Marks', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Final Marks', fontsize=14)
plt.axvline(df_processed['Final_Marks'].mean(), color='red', linestyle='--',
            label=f'Mean: {df_processed["Final_Marks"].mean():.1f}')
plt.axvline(df_processed['Final_Marks'].median(), color='green', linestyle='--',
            label=f'Median: {df_processed["Final_Marks"].median():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(y=df_processed['Final_Marks'], color='lightgreen')
plt.ylabel('Final Marks', fontsize=12)
plt.title('Box Plot of Final Marks', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/distribution_final_marks.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean Final Marks: {df_processed['Final_Marks'].mean():.2f}")
print(f"Median Final Marks: {df_processed['Final_Marks'].median():.2f}")
print(f"Standard Deviation: {df_processed['Final_Marks'].std():.2f}")

# 4.2 Study Hours vs Final Marks
print("\n--- 4.2 Study Hours vs Final Marks ---")

plt.figure(figsize=(10, 6))
plt.scatter(df_processed['Study_Hours'], df_processed['Final_Marks'],
            alpha=0.6, c='steelblue', edgecolors='white', s=50)
plt.xlabel('Study Hours per Day', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Study Hours vs Final Marks', fontsize=14)

# Add trend line
z = np.polyfit(df_processed['Study_Hours'], df_processed['Final_Marks'], 1)
p = np.poly1d(z)
plt.plot(sorted(df_processed['Study_Hours']), p(sorted(df_processed['Study_Hours'])),
         "r--", alpha=0.8, label=f'Trend Line (r={np.corrcoef(df_processed["Study_Hours"], df_processed["Final_Marks"])[0,1]:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/study_hours_vs_marks.png', dpi=150, bbox_inches='tight')
plt.show()

correlation_study = df_processed['Study_Hours'].corr(df_processed['Final_Marks'])
print(f"Correlation between Study Hours and Final Marks: {correlation_study:.3f}")

# 4.3 Attendance Percentage vs Final Marks
print("\n--- 4.3 Attendance vs Final Marks ---")

plt.figure(figsize=(10, 6))
plt.scatter(df_processed['Attendance_Percentage'], df_processed['Final_Marks'],
            alpha=0.6, c='coral', edgecolors='white', s=50)
plt.xlabel('Attendance Percentage', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Attendance Percentage vs Final Marks', fontsize=14)

# Add trend line
z = np.polyfit(df_processed['Attendance_Percentage'], df_processed['Final_Marks'], 1)
p = np.poly1d(z)
plt.plot(sorted(df_processed['Attendance_Percentage']), p(sorted(df_processed['Attendance_Percentage'])),
         "r--", alpha=0.8, label=f'Trend Line (r={np.corrcoef(df_processed["Attendance_Percentage"], df_processed["Final_Marks"])[0,1]:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/attendance_vs_marks.png', dpi=150, bbox_inches='tight')
plt.show()

correlation_attendance = df_processed['Attendance_Percentage'].corr(df_processed['Final_Marks'])
print(f"Correlation between Attendance and Final Marks: {correlation_attendance:.3f}")

# 4.4 Sleep Duration vs Final Marks
print("\n--- 4.4 Sleep Duration vs Final Marks ---")

plt.figure(figsize=(10, 6))
plt.scatter(df_processed['Sleep_Duration'], df_processed['Final_Marks'],
            alpha=0.6, c='mediumpurple', edgecolors='white', s=50)
plt.xlabel('Sleep Duration (hours)', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Sleep Duration vs Final Marks', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('plots/sleep_vs_marks.png', dpi=150, bbox_inches='tight')
plt.show()

correlation_sleep = df_processed['Sleep_Duration'].corr(df_processed['Final_Marks'])
print(f"Correlation between Sleep Duration and Final Marks: {correlation_sleep:.3f}")

# 4.5 Previous Scores vs Final Marks
print("\n--- 4.5 Previous Scores vs Final Marks ---")

plt.figure(figsize=(10, 6))
plt.scatter(df_processed['Previous_Scores'], df_processed['Final_Marks'],
            alpha=0.6, c='gold', edgecolors='white', s=50)
plt.xlabel('Previous Scores', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Previous Scores vs Final Marks', fontsize=14)

# Add trend line
z = np.polyfit(df_processed['Previous_Scores'], df_processed['Final_Marks'], 1)
p = np.poly1d(z)
plt.plot(sorted(df_processed['Previous_Scores']), p(sorted(df_processed['Previous_Scores'])),
         "r--", alpha=0.8, label=f'Trend Line (r={np.corrcoef(df_processed["Previous_Scores"], df_processed["Final_Marks"])[0,1]:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/previous_scores_vs_marks.png', dpi=150, bbox_inches='tight')
plt.show()

correlation_previous = df_processed['Previous_Scores'].corr(df_processed['Final_Marks'])
print(f"Correlation between Previous Scores and Final Marks: {correlation_previous:.3f}")

# 4.6 Assignment Scores vs Final Marks
print("\n--- 4.6 Assignment Scores vs Final Marks ---")

plt.figure(figsize=(10, 6))
plt.scatter(df_processed['Assignment_Scores'], df_processed['Final_Marks'],
            alpha=0.6, c='lightcoral', edgecolors='white', s=50)
plt.xlabel('Assignment Scores', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Assignment Scores vs Final Marks', fontsize=14)

# Add trend line
z = np.polyfit(df_processed['Assignment_Scores'], df_processed['Final_Marks'], 1)
p = np.poly1d(z)
plt.plot(sorted(df_processed['Assignment_Scores']), p(sorted(df_processed['Assignment_Scores'])),
         "r--", alpha=0.8, label=f'Trend Line (r={np.corrcoef(df_processed["Assignment_Scores"], df_processed["Final_Marks"])[0,1]:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/assignment_scores_vs_marks.png', dpi=150, bbox_inches='tight')
plt.show()

correlation_assignment = df_processed['Assignment_Scores'].corr(df_processed['Final_Marks'])
print(f"Correlation between Assignment Scores and Final Marks: {correlation_assignment:.3f}")

# 4.7 Correlation Heatmap
print("\n--- 4.7 Correlation Heatmap ---")

plt.figure(figsize=(12, 10))
numeric_df = df_processed.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.2f', cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of All Numeric Features', fontsize=14)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 features most correlated with Final Marks:")
target_corr = correlation_matrix['Final_Marks'].drop('Final_Marks').sort_values(key=abs, ascending=False)
for i, (feat, corr) in enumerate(target_corr.head(5).items(), 1):
    print(f"  {i}. {feat}: {corr:.3f}")

# 4.8 Gender-wise Performance
print("\n--- 4.8 Gender-wise Performance ---")

plt.figure(figsize=(10, 6))
gender_performance = df_processed.groupby('Gender')['Final_Marks'].agg(['mean', 'median', 'std'])
print("\nGender-wise Performance Statistics:")
print(gender_performance.round(2))

gender_counts = df_processed['Gender'].value_counts()
plt.subplot(1, 2, 1)
plt.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'pink'])
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Gender Distribution', fontsize=14)

plt.subplot(1, 2, 2)
sns.boxplot(x='Gender', y='Final_Marks', data=df_processed, palette=['lightblue', 'pink'])
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Final Marks by Gender', fontsize=14)

plt.tight_layout()
plt.savefig('plots/gender_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.9 Stream-wise Performance
print("\n--- 4.9 Stream-wise Performance ---")

plt.figure(figsize=(12, 6))
stream_performance = df_processed.groupby('Stream')['Final_Marks'].agg(['mean', 'median', 'std'])
print("\nStream-wise Performance Statistics:")
print(stream_performance.round(2))

plt.subplot(1, 2, 1)
stream_counts = df_processed['Stream'].value_counts()
plt.pie(stream_counts.values, labels=stream_counts.index, autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Stream Distribution', fontsize=14)

plt.subplot(1, 2, 2)
sns.boxplot(x='Stream', y='Final_Marks', data=df_processed,
            order=['Science', 'Commerce', 'Arts'], palette=['#ff9999', '#66b3ff', '#99ff99'])
plt.xlabel('Stream', fontsize=12)
plt.ylabel('Final Marks', fontsize=12)
plt.title('Final Marks by Stream', fontsize=14)

plt.tight_layout()
plt.savefig('plots/stream_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# 4.10 Parental Education Impact
print("\n--- 4.10 Parental Education Impact ---")

plt.figure(figsize=(10, 6))
education_performance = df_processed.groupby('Parental_Education_Label')['Final_Marks'].mean()
print("\nParental Education vs Average Final Marks:")
print(education_performance.round(2))

plt.bar(education_performance.index, education_performance.values,
        color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.xlabel('Parental Education Level', fontsize=12)
plt.ylabel('Average Final Marks', fontsize=12)
plt.title('Impact of Parental Education on Student Performance', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('plots/parental_education_impact.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. MODEL BUILDING
# =============================================================================

print("\n" + "=" * 80)
print("5. MODEL BUILDING - LINEAR REGRESSION")
print("=" * 80)

# 5.1 Split the dataset
print("\n--- 5.1 Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                     random_state=42)

print(f"Training set size: {X_train.shape[0]} samples ({int(X_train.shape[0]/len(X_scaled)*100)}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({int(X_test.shape[0]/len(X_scaled)*100)}%)")

# 5.2 Train the Linear Regression model
print("\n--- 5.2 Training Linear Regression Model ---")

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# 5.3 Display model coefficients
print("\n--- 5.3 Model Coefficients ---")

print(f"\nIntercept (Bias): {model.intercept_:.4f}")
print("\nCoefficients for each feature:")

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for _, row in coef_df.iterrows():
    print(f"  {row['Feature']:30s}: {row['Coefficient']:+.4f}")

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')
print("\nModel saved as 'linear_regression_model.pkl'")

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("6. MODEL EVALUATION")
print("=" * 80)

# 6.1 Make predictions
print("\n--- 6.1 Making Predictions ---")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Training predictions generated: {len(y_train_pred)}")
print(f"Testing predictions generated: {len(y_test_pred)}")

# 6.2 Calculate evaluation metrics
print("\n--- 6.2 Evaluation Metrics ---")

# Training set metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Testing set metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "-" * 40)
print("TRAINING SET PERFORMANCE:")
print("-" * 40)
print(f"  Mean Absolute Error (MAE):  {train_mae:.4f}")
print(f"  Mean Squared Error (MSE):   {train_mse:.4f}")
print(f"  Root Mean Squared Error:    {train_rmse:.4f}")
print(f"  R² Score:                   {train_r2:.4f}")

print("\n" + "-" * 40)
print("TESTING SET PERFORMANCE:")
print("-" * 40)
print(f"  Mean Absolute Error (MAE):  {test_mae:.4f}")
print(f"  Mean Squared Error (MSE):   {test_mse:.4f}")
print(f"  Root Mean Squared Error:    {test_rmse:.4f}")
print(f"  R² Score:                   {test_r2:.4f}")

# 6.3 Metric interpretation
print("\n--- 6.3 Metric Interpretation ---")

print(f"""
MAE (Mean Absolute Error): {test_mae:.2f}
  -> On average, predictions are off by ±{test_mae:.2f} marks

MSE (Mean Squared Error): {test_mse:.2f}
  -> Penalizes larger errors more heavily

RMSE (Root Mean Squared Error): {test_rmse:.2f}
  -> Average prediction error in the same units as the target

R² Score: {test_r2:.4f}
  -> The model explains {test_r2*100:.2f}% of the variance in Final Marks
""")

# =============================================================================
# 7. RESULTS VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("7. RESULTS VISUALIZATION")
print("=" * 80)

# 7.1 Predicted vs Actual Values
print("\n--- 7.1 Predicted vs Actual Values ---")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, c='steelblue', edgecolors='white', s=50)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values (Training)', fontsize=12)
plt.ylabel('Predicted Values (Training)', fontsize=12)
plt.title('Training Set: Predicted vs Actual', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, c='coral', edgecolors='white', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values (Testing)', fontsize=12)
plt.ylabel('Predicted Values (Testing)', fontsize=12)
plt.title('Testing Set: Predicted vs Actual', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()

# 7.2 Residual Analysis
print("\n--- 7.2 Residual Analysis ---")

residuals = y_test - y_test_pred

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals, alpha=0.6, c='mediumpurple', edgecolors='white', s=50)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(residuals, bins=20, kde=True, color='skyblue')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals', fontsize=14)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Residual Statistics:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std Dev: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# 7.3 Feature Importance (based on coefficient magnitude)
print("\n--- 7.3 Feature Importance ---")

plt.figure(figsize=(10, 8))
coef_df_sorted = coef_df.sort_values('Coefficient', key=abs, ascending=True)
colors = ['#ff9999' if x < 0 else '#66b3ff' for x in coef_df_sorted['Coefficient']]
plt.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'], color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance (Linear Regression Coefficients)', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFeature Importance Ranking (by absolute coefficient value):")
for i, (_, row) in enumerate(coef_df.sort_values('Coefficient', key=abs, ascending=False).iterrows(), 1):
    direction = "positive" if row['Coefficient'] > 0 else "negative"
    print(f"  {i}. {row['Feature']:30s}: {row['Coefficient']:+.4f} ({direction})")

# =============================================================================
# 8. CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("8. CONCLUSION")
print("=" * 80)

conclusion = f"""
SUMMARY OF FINDINGS:
================================================================================

1. MODEL PERFORMANCE:
   - The Linear Regression model achieved an R² score of {test_r2:.4f} on the test set.
   - This means the model explains approximately {test_r2*100:.1f}% of the variance
     in student final marks.
   - Mean Absolute Error (MAE): {test_mae:.2f} marks
   - Root Mean Squared Error (RMSE): {test_rmse:.2f} marks

2. KEY FACTORS AFFECTING STUDENT PERFORMANCE:
   Based on the model coefficients, the most influential factors are:
"""

print(conclusion)

for i, (_, row) in enumerate(coef_df.sort_values('Coefficient', key=abs, ascending=False).head(5).iterrows(), 1):
    direction = "positively" if row['Coefficient'] > 0 else "negatively"
    print(f"   {i}. {row['Feature']}: {direction} correlated with performance")

conclusion2 = f"""

3. INSIGHTS:
   - Previous academic performance (Previous Scores, Assignment Scores) are
     strong predictors of final performance.
   - Study hours show a positive correlation with marks, emphasizing the
     importance of consistent study habits.
   - Attendance percentage is positively associated with better performance.
   - Sleep duration has an optimal range; both too little and too much sleep
     can affect performance.
   - Distance from school has a slight negative impact on performance.

4. LIMITATIONS:
   - The model assumes linear relationships between features and target.
   - Some factors like motivation, learning style, and teaching quality are
     not captured in the dataset.
   - The dataset is synthetically generated and may not fully represent
     real-world complexities.

5. RECOMMENDATIONS:
   - Students should maintain regular attendance and consistent study schedules.
   - Ensuring adequate sleep (7-8 hours) can improve academic performance.
   - Educators can use this model to identify at-risk students early.
   - Schools should consider providing support for students with long
     commute distances.

================================================================================
PROJECT FILES GENERATED:
================================================================================
- student_performance_raw.csv         : Raw dataset
- student_performance_processed.csv   : Preprocessed dataset
- linear_regression_model.pkl         : Trained model
- scaler.pkl                          : Feature scaler
- label_encoder_gender.pkl            : Gender label encoder
- label_encoder_stream.pkl            : Stream label encoder
- plots/                              : Directory containing all visualizations

================================================================================
"""
print(conclusion2)

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)