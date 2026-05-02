import numpy as np
import pandas as pd

np.random.seed(123)

N = 10000

# -------------------------
# Base Features (same schema)
# -------------------------
age = np.random.randint(15, 76, N)
gender = np.random.choice(["male", "female", "other"], N, p=[0.48, 0.48, 0.04])
weight = np.round(np.random.uniform(45.0, 140.0, N), 1)
height = np.round(np.random.uniform(1.50, 2.05, N), 2)
bmi = np.round(weight / (height ** 2), 2)

goal = np.random.choice(
    ["fat loss", "muscle gain", "endurance", "general fitness"], 
    N, 
    p=[0.28, 0.28, 0.22, 0.22]
)

health_condition = np.random.choice(
    ["none", "asthma", "injury", "hypertension", "heart disease", "diabetes", "pcos"],
    N,
    p=[0.62, 0.08, 0.08, 0.08, 0.05, 0.05, 0.04]
)

activity_level = np.random.choice(["low", "moderate", "high"], N, p=[0.34, 0.38, 0.28])
experience_level = np.random.choice(["beginner", "intermediate", "advanced"], N, p=[0.38, 0.42, 0.20])
dietary_preference = np.random.choice(
    ["veg", "non-veg", "vegan", "pescatarian"], 
    N
)

sleep_hours = np.round(np.random.uniform(4.5, 9.5, N), 1)
workout_type_preference = np.random.choice(
    ["hiit", "cardio", "strength", "flexibility", "balance", "functional"], 
    N
)

equipment_available_count = np.random.randint(0, 7, N)
time_available = np.random.randint(15, 121, N)

has_allergy = np.random.choice([0, 1], N, p=[0.8, 0.2])
has_digestive_issue = np.random.choice([0, 1], N, p=[0.85, 0.15])
smoking = np.random.choice([0, 1], N, p=[0.78, 0.22])
cigarettes_per_day = np.where(smoking == 1, np.random.uniform(1, 18, N), 0)
cigarettes_per_day = np.round(cigarettes_per_day, 1)

alcohol = np.random.choice([0, 1], N, p=[0.72, 0.28])
alcohol_units_per_week = np.where(alcohol == 1, np.random.uniform(1, 25, N), 0)
alcohol_units_per_week = np.round(alcohol_units_per_week, 1)

micronutrient_score = np.round(np.random.uniform(0.0, 1.0, N), 2)

# Extra helpful signals
protein_intake_g = np.round(np.random.uniform(40, 220, N), 1)
steps_per_day = np.random.randint(1500, 18000, N)
water_intake_liters = np.round(np.random.uniform(1.0, 4.5, N), 2)

# -------------------------
# STRONGER Target Logic (4 classes)
# -------------------------

def workout_level(row):
    score = 0
    
    if row["activity_level"] == "high": score += 2
    if row["experience_level"] == "advanced": score += 2
    if row["time_available"] > 75: score += 1
    if row["goal"] == "muscle gain": score += 1
    if row["bmi"] > 32: score += 1
    
    # Clear separation bands
    if score <= 1:
        return "None"
    elif score == 2:
        return "Low"
    elif score in [3,4]:
        return "Medium"
    else:
        return "High"

def nutrition_level(row):
    score = 0
    
    if row["goal"] in ["fat loss", "muscle gain"]: score += 2
    if row["bmi"] > 29: score += 1
    if row["micronutrient_score"] < 0.4: score += 1
    if row["protein_intake_g"] < 70: score += 1
    if row["alcohol"] == 1: score += 1
    
    if score <= 1:
        return "None"
    elif score == 2:
        return "Low"
    elif score in [3,4]:
        return "Medium"
    else:
        return "High"

df2 = pd.DataFrame({
    "age": age,
    "gender": gender,
    "weight": weight,
    "height": height,
    "bmi": bmi,
    "goal": goal,
    "health_condition": health_condition,
    "activity_level": activity_level,
    "experience_level": experience_level,
    "dietary_preference": dietary_preference,
    "sleep_hours": sleep_hours,
    "workout_type_preference": workout_type_preference,
    "equipment_available_count": equipment_available_count,
    "time_available": time_available,
    "has_allergy": has_allergy,
    "has_digestive_issue": has_digestive_issue,
    "smoking": smoking,
    "cigarettes_per_day": cigarettes_per_day,
    "alcohol": alcohol,
    "alcohol_units_per_week": alcohol_units_per_week,
    "micronutrient_score": micronutrient_score,
    "protein_intake_g": protein_intake_g,
    "steps_per_day": steps_per_day,
    "water_intake_liters": water_intake_liters
})

df2["workout_plan_level"] = df2.apply(workout_level, axis=1)
df2["nutrition_plan_level"] = df2.apply(nutrition_level, axis=1)

# Save new version
file_path2 = "Dataset/fitness_goals.csv"
df2.to_csv(file_path2, index=False)

df2["workout_plan_level"].value_counts(), df2["nutrition_plan_level"].value_counts(), file_path2