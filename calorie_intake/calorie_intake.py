import pandas as pd
import dill
import os
import tensorflow as tf

def transform_data(
    age: int,
    gender: str, # 'F' or 'M'
    daily_calories_consumed: int,
    weight_change_in_lbs: float,
    duration_in_weeks: float,
    physical_activity_level: str, # 'Active', 'Moderately Active', 'Sedentary', 'Very Active'
    sleep_quality: str, # 'Poor', 'Fair', 'Good', 'Excellent'
    stress_level: int,
    current_weight_in_lbs: float,
    caloric_adjustment: float
) -> pd.DataFrame:
    transformed_data = {
        'Age': age,
        'Current Weight (lbs)': current_weight_in_lbs,
        'Daily Calories Consumed': daily_calories_consumed,
        'Weight Change (lbs)': weight_change_in_lbs,
        'Duration (weeks)': duration_in_weeks,
        'Stress Level': stress_level,
        'Caloric Adjustment': caloric_adjustment,
        'Gender_F': 1 if gender == 'F' else 0,
        'Gender_M': 1 if gender == 'M' else 0,
        'Physical Activity Level_Lightly Active': 1 if physical_activity_level == 'Active' else 0,
        'Physical Activity Level_Moderately Active': 1 if physical_activity_level == 'Moderately Active' else 0,
        'Physical Activity Level_Sedentary': 1 if physical_activity_level == 'Sedentary' else 0,
        'Physical Activity Level_Very Active': 1 if physical_activity_level == 'Very Active' else 0,
        'Sleep Quality_Excellent': 1 if sleep_quality == 'Excellent' else 0,
        'Sleep Quality_Fair': 1 if sleep_quality == 'Fair' else 0,
        'Sleep Quality_Good': 1 if sleep_quality == 'Good' else 0,
        'Sleep Quality_Poor': 1 if sleep_quality == 'Poor' else 0,
    }

    return pd.DataFrame([transformed_data])

with open(os.path.join('calorie_intake', 'model', 'pipeline.pkl'), 'rb') as f:
    pipeline = dill.load(f)

def transform_pipeline(transformed_data_df: pd.DataFrame) -> pd.DataFrame:
    result = pipeline.transform(transformed_data_df)

    return result

model = tf.keras.models.load_model(os.path.join('calorie_intake', 'model', 'exp1.h5'))

def predict_calorie_intake(processed_data_df: pd.DataFrame) -> float:
    result = model.predict(processed_data_df)

    return float(result[0][0])

def full_prediction_pipeline(
    age: int,
    gender: str, # 'F' or 'M'
    daily_calories_consumed: float,
    weight_change_in_lbs: float,
    duration_in_weeks: float,
    physical_activity_level: str, # 'Active', 'Moderately Active', 'Sedentary', 'Very Active'
    sleep_quality: str, # 'Poor', 'Fair', 'Good', 'Excellent'
    stress_level: int,
    current_weight_in_lbs: float,
    caloric_adjustment: int
) -> float:
    transformed_data = transform_data(
        age,
        gender,
        daily_calories_consumed,
        weight_change_in_lbs,
        duration_in_weeks,
        physical_activity_level,
        sleep_quality,
        stress_level,
        current_weight_in_lbs,
        caloric_adjustment
    )

    processed_data = transform_pipeline(transformed_data)

    result = predict_calorie_intake(processed_data)

    return result
