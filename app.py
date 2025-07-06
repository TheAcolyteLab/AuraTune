import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import joblib
import streamlit as st

model = joblib.load('model/music_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_order = joblib.load('model/feature_order.pkl') 

print("=== AuraTune : MUSIC THERAPY EFFECTIVENESS PREDICTOR ===\n")

# Load the dataset
df = pd.read_csv('data/mxmh_survey_results.csv')
print(f"Dataset loaded with shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Data exploration
print("\n=== DATA EXPLORATION ===")

#target variable
target_col = "Music effects"

if target_col in df.columns:
    print(f"Using target column: {target_col}")
    print(f"Unique values: {df[target_col].unique()}")

# Create binary target variable (adjust based on actual column values)

    df['music_helps_therapy'] = df[target_col].apply(
        lambda x: 1 if str(x).lower() in ['improve', 'significantly improve', 'help', 'better'] else 0
    )
else:
    print("Target column 'Music effects' not found. Creating demo target variable...")
    np.random.seed(42)
    df['music_helps_therapy'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])

print(f"\nTarget variable distribution:")
print(df['music_helps_therapy'].value_counts())
print(f"Percentage who find music helpful: {df['music_helps_therapy'].mean():.2%}")


print("\n=== FEATURE ENGINEERING ===")

# Use the correct column names from the dataset
age_col = "Age"
hours_col = "Hours per day"
genre_col = "Fav genre"
platform_col = "Primary streaming service"

print(f"Using columns - Age: {age_col}, Hours: {hours_col}, Genre: {genre_col}, Platform: {platform_col}")

# Create age-related features
if age_col in df.columns:
    df['age_numeric'] = pd.to_numeric(df[age_col], errors='coerce')
    # Fill missing values with median of non-null values
    median_age = df['age_numeric'].median()
    df['age_numeric'] = df['age_numeric'].fillna(median_age)
else:
    np.random.seed(42)
    df['age_numeric'] = np.random.normal(28, 8, len(df))

df['young_adult'] = (df['age_numeric'] <= 25).astype(int)
df['millennial'] = ((df['age_numeric'] > 25) & (df['age_numeric'] <= 40)).astype(int)
df['older_adult'] = (df['age_numeric'] > 40).astype(int)

# Create listening intensity categories
if hours_col in df.columns:
    # Convert hours to numeric, handling potential string values
    df['hours_numeric'] = pd.to_numeric(df[hours_col], errors='coerce')
    # Fill missing values with median of non-null values
    median_hours = df['hours_numeric'].median()
    df['hours_numeric'] = df['hours_numeric'].fillna(median_hours)
else:
    np.random.seed(43)  
    df['hours_numeric'] = np.random.exponential(2.5, len(df))

# df['light_listener'] = (df['hours_numeric'] < 2).astype(int)
df['moderate_listener'] = ((df['hours_numeric'] >= 2) & (df['hours_numeric'] <= 4)).astype(int)
df['heavy_listener'] = (df['hours_numeric'] > 4).astype(int)


df['heavy_listener'] = (df['hours_numeric'] > 4).astype(int)
# Create genre preferences (adjust based on actual genre values)
if genre_col in df.columns:
    print(f"Available genres: {df[genre_col].unique()}")
    
    # Define genre categories based on the typical genres in the dataset
    calming_genres = ['Classical', 'Jazz', 'Folk', 'Country', 'Blues', 'Gospel', 'Lofi']
    energizing_genres = ['Rock', 'Metal', 'Hip hop', 'EDM', 'Pop', 'Rap', 'K pop']
    
    df['prefers_calming'] = df[genre_col].apply(
        lambda x: 1 if any(genre.lower() in str(x).lower() for genre in calming_genres) else 0
    )
    df['prefers_energizing'] = df[genre_col].apply(
        lambda x: 1 if any(genre.lower() in str(x).lower() for genre in energizing_genres) else 0
    )
else:
    np.random.seed(44)
    df['prefers_calming'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    df['prefers_energizing'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
# Platform preferences
if platform_col in df.columns:
    print(f"Available platforms: {df[platform_col].unique()}")
    df['uses_spotify'] = df[platform_col].apply(lambda x: 1 if 'spotify' in str(x).lower() else 0)
    df['uses_apple_music'] = df[platform_col].apply(lambda x: 1 if 'apple' in str(x).lower() else 0)
    df['uses_youtube'] = df[platform_col].apply(lambda x: 1 if 'youtube' in str(x).lower() else 0)
else:
    np.random.seed(45)
    df['uses_spotify'] = np.random.choice([0, 1], len(df), p=[0.4, 0.6])
    df['uses_apple_music'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    df['uses_youtube'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])

# Mental health indicators (adjust based on actual columns)
mental_health_cols = ['Anxiety','Depression','Insomnia','OCD']
existing_mental_health_cols = []

for col in mental_health_cols:
    if col in df.columns:
        existing_mental_health_cols.append(col)
        print(f"Found mental health column: {col} with values: {df[col].unique()}")
        
    
        # Convert to binary format
        # Assuming values might be like 'Yes'/'No' or scale values
        if df[col].dtype == 'object':
            df[col + '_binary'] = df[col].apply(
                lambda x: 1 if str(x).lower() in ['yes', 'true', 'severe', 'moderate'] else 0
            )
        else:
            # If numeric, assume higher values mean presence of condition
            df[col + '_binary'] = (df[col] > 0).astype(int)
            
        # Use the binary version
        df[col] = df[col + '_binary']

if not existing_mental_health_cols:
    print("No mental health columns found in expected format. Creating demo data...")
    # Create demo mental health data with different seeds
    np.random.seed(46)
    df['Anxiety'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
    np.random.seed(47)
    df['Depression'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    np.random.seed(48)
    df['Insomnia'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    existing_mental_health_cols = ['Anxiety', 'Depression', 'Insomnia']

print(f"Using mental health columns: {existing_mental_health_cols}")

# Create composite mental health score

valid_mental_health_cols = [col for col in existing_mental_health_cols if col in df.columns]
if valid_mental_health_cols:
    df['mental_health_score'] = df[valid_mental_health_cols].sum(axis=1)
else:
    df['mental_health_score'] = 0
    
df['multiple_conditions'] = (df['mental_health_score'] >= 2).astype(int)

# Add frequency-based features from the dataset
frequency_cols = [col for col in df.columns if col.startswith('Frequency')]
if frequency_cols:
    print(f"Found frequency columns: {frequency_cols[:5]}...")  # Show first 5
    
    # Create features for music genre diversity
    # Convert frequency responses to numeric
    freq_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    
    for col in frequency_cols:
        df[col + '_numeric'] = df[col].map(freq_mapping).fillna(0)
    
    # Calculate genre diversity (how many genres they listen to frequently)
    freq_numeric_cols = [col + '_numeric' for col in frequency_cols]
    df['genre_diversity'] = (df[freq_numeric_cols] >= 2).sum(axis=1)  # Count genres listened to sometimes or more
    df['high_genre_diversity'] = (df['genre_diversity'] >= 5).astype(int)
    
    # Specific genre preferences
    if 'Frequency [Classical]_numeric' in df.columns:
        df['classical_listener'] = (df['Frequency [Classical]_numeric'] >= 2).astype(int)
    if 'Frequency [Rock]_numeric' in df.columns:
        df['rock_listener'] = (df['Frequency [Rock]_numeric'] >= 2).astype(int)
    if 'Frequency [Jazz]_numeric' in df.columns:
        df['jazz_listener'] = (df['Frequency [Jazz]_numeric'] >= 2).astype(int)

print("Feature engineering completed!")

# MODEL BUILDING
print("\n=== MODEL BUILDING ===")

# Select features for the model (updated with new features)
potential_features = [
    'age_numeric', 'hours_numeric', 'prefers_calming', 'prefers_energizing',
    'moderate_listener', 'heavy_listener', 'uses_spotify', 'uses_apple_music', 'uses_youtube',
    'mental_health_score', 'multiple_conditions', 'millennial', 'young_adult',
    'genre_diversity', 'high_genre_diversity', 'classical_listener', 'rock_listener', 'jazz_listener',
    'Anxiety', 'Depression', 'Insomnia', 'OCD'
]

# Filter features that actually exist in the dataframe
available_features = [col for col in potential_features if col in df.columns]
print(f"Available features for modeling: {available_features}")

if not available_features:
    print("No features available for modeling. Please check feature engineering.")
    exit()

# Prepare data
X = df[available_features].copy()
y = df['music_helps_therapy'].copy()

# Handle missing values more carefully
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

# Remove any rows where target is missing
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"Final dataset for modeling: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Check if we have enough data
if len(X) < 10:
    print("Not enough data for modeling. Please check your dataset.")
    exit()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# MODEL EVALUATION
print("\n=== MODEL EVALUATION ===")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC Score: {auc_score:.3f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Music Therapy Effectiveness')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# FEATURE IMPORTANCE ANALYSIS
print("\n=== FEATURE IMPORTANCE ===")

# Get feature importance from logistic regression coefficients
feature_importance = pd.DataFrame({
    'feature': available_features,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("Feature Importance (sorted by absolute coefficient value):")
print(feature_importance)

# Plot feature importance


# INSIGHTS AND INTERPRETATION
print("\n=== KEY INSIGHTS ===")

top_positive = feature_importance[feature_importance['coefficient'] > 0].head(3)
top_negative = feature_importance[feature_importance['coefficient'] < 0].head(3)

print("Factors that INCREASE likelihood of music therapy effectiveness:")
for _, row in top_positive.iterrows():
    print(f"  • {row['feature']}: coefficient = {row['coefficient']:.3f}")

print("\nFactors that DECREASE likelihood of music therapy effectiveness:")
for _, row in top_negative.iterrows():
    print(f"  • {row['feature']}: coefficient = {row['coefficient']:.3f}")

# Sample predictions for interpretation
print(f"\n=== SAMPLE PREDICTIONS ===")
if len(X_test) > 0:
    n_samples = min(5, len(X_test))
    sample_indices = np.random.choice(X_test.index, n_samples, replace=False)
    
    for idx in sample_indices:
        sample_features = X_test.loc[idx]
        sample_scaled = scaler.transform([sample_features])
        sample_prediction = model.predict_proba(sample_scaled)[0]
        actual_value = y_test.loc[idx]
        
        print(f"\nSample {idx}:")
        print(f"  Actual music helps: {'Yes' if actual_value == 1 else 'No'}")
        print(f"  Predicted probability: {sample_prediction[1]:.3f}")
        if 'age_numeric' in sample_features.index and 'hours_numeric' in sample_features.index:
            print(f"  Key features: Age={sample_features['age_numeric']:.1f}, Hours={sample_features['hours_numeric']:.1f}")

# Save model and scaler
try:
    joblib.dump(model, 'model/music_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'model/feature_order.pkl')
    print("\n=== MODEL SAVED SUCCESSFULLY ===")
except Exception as e:
    print(f"Error saving model: {e}")

print("\n=== MODEL READY FOR USE ===")


st.set_page_config(page_title="AuraTune 🎵", layout="centered")
st.title("🎧 AuraTune: Music Therapy Effectiveness Predictor")
st.markdown("Predict if music can help you mentally, based on your habits and health background.")

st.subheader("📝 Enter Your Details")

# User Inputs
age = st.slider("Your Age", 10, 80, 25)
hours = st.slider("Hours of Music per Day", 0.0, 10.0, 2.0, step=0.5)
genre = st.selectbox("Favorite Genre", ['Pop', 'Rock', 'Jazz', 'Classical', 'EDM', 'Hip hop', 'Metal', 'Country', 'Lofi','KPop','Folk'])
platform = st.selectbox("Primary Streaming Platform", ['Spotify', 'Apple Music', 'YouTube', 'Other'])

st.subheader("🧠 Mental Health Conditions")
anxiety = st.checkbox("Anxiety")
depression = st.checkbox("Depression")
insomnia = st.checkbox("Insomnia")
ocd = st.checkbox("OCD")

# Feature engineering (match your model input features)
input_features = {}

input_features['age_numeric'] = age
input_features['hours_numeric'] = hours

# Listener category
input_features['moderate_listener'] = int(1 >= hours <= 4)
input_features['heavy_listener'] = int(hours > 4)

# Genre preferences
calming_genres = ['Classical', 'Jazz', 'Folk', 'Country', 'Blues', 'Gospel', 'Lofi']
energizing_genres = ['Rock', 'Metal', 'Hip hop', 'EDM', 'Pop', 'Rap', 'K pop']

input_features['prefers_calming'] = int(genre in calming_genres)
input_features['prefers_energizing'] = int(genre in energizing_genres)

# Platform usage
input_features['uses_spotify'] = int(platform == 'Spotify')
input_features['uses_apple_music'] = int(platform == 'Apple Music')
input_features['uses_youtube'] = int(platform == 'YouTube')

# Age group
input_features['young_adult'] = int(age <= 25)
input_features['millennial'] = int(25 < age <= 40)

# Mental health binary
input_features['Anxiety'] = int(anxiety)
input_features['Depression'] = int(depression)
input_features['Insomnia'] = int(insomnia)
input_features['OCD'] = int(ocd)

# Mental health derived
mental_health_score = (
    input_features['Anxiety'] + input_features['Depression'] +
    input_features['Insomnia'] + input_features['OCD']
)
input_features['mental_health_score'] = mental_health_score
input_features['multiple_conditions'] = int(mental_health_score >= 2)

# Genre diversity placeholders
input_features['genre_diversity'] = 5 
input_features['high_genre_diversity'] = int(input_features['genre_diversity'] >= 5)

# Popular genres placeholder
input_features['classical_listener'] = int(genre == 'Classical')
input_features['rock_listener'] = int(genre == 'Rock')
input_features['jazz_listener'] = int(genre == 'Jazz')

# Convert to DataFrame and scale

X_input = pd.DataFrame([input_features])
X_input = X_input[feature_order]
X_scaled = scaler.transform(X_input)

# Predict
if st.button("🎶 Predict"):
    proba = model.predict_proba(X_scaled)[0][1]
    result = "✅ Music is likely beneficial for you!" if proba >= 0.6 else "❌ Music might not be very helpful as therapy."

    st.subheader("🧾 Result")
    st.metric("Probability", f"{proba:.2%}")
    st.success(result if proba >= 0.6 else result)