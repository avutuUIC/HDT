import pickle
import pandas as pd
from multi_task_model import MultiTaskModel
import joblib

# Load all models and encoders
# with open("models/anxiety_model.pkl", "rb") as f:
#     anxiety_model = pickle.load(f)
anxiety_model = joblib.load("models/anxiety_model.pkl")
# with open("models/stress_model.pkl", "rb") as f:
#     stress_model = pickle.load(f)
stress_model = joblib.load("models/stress_model.pkl")
# with open("models/motivation_model.pkl", "rb") as f:
#     motivation_model = pickle.load(f)
motivation_model = joblib.load("models/motivation_model.pkl")
with open("models/physicalModel.pkl", "rb") as f:
    physical_package = pickle.load(f)

cognitive_model = joblib.load("models/multi_label_cognitive_model.pkl")
cognitive_features = joblib.load("models/model_features.pkl")
balance_model = joblib.load("models/model_balance.pkl")
dexterity_model = joblib.load("models/model_dexterity.pkl")
posture_model = joblib.load("models/model_posture.pkl")
leader_model = joblib.load("models/leadership_predict.pkl")
engage_model = joblib.load("models/engagement_predict.pkl")
engage_encoder = joblib.load("models/engagement_encoder.pkl")
coop_model = joblib.load("models/cooperation_predict.pkl")

# Optional: feature lists to avoid hardcoding everywhere
MENTAL_FEATURES = [
    "eda", "eda_delta", "hrv_mean_hr", "hrv_sdnn", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf"
]

COGNITIVE_FEATURES = [
    "time_on_task", "num_correct", "num_incorrect", "hr_mean", "hr_std", "gsr_mean", "gsr_std",
    "rr_mean", "rr_std", "temperature_mean", "temperature_std",
    "band_ax_mean", "band_ax_std", "band_ay_mean", "band_ay_std", "band_az_mean", "band_az_std"
]

BIOMECHANICAL_FEATURES = [
    "x_mean","x_std","x_min","x_max","x_range","x_median","x_iqr",
    "y_mean","y_std","y_min","y_max","y_range","y_median","y_iqr",
    "z_mean","z_std","z_min","z_max","z_range","z_median","z_iqr",
    "magnitude_mean","magnitude_std","magnitude_min","magnitude_max","magnitude_range",
    "magnitude_median","magnitude_iqr","jerk_mean","jerk_std","total_movement"
]

INTERPERSONAL_FEATURES = [
    "avg_power", "total_speaking_time", "initiations", "avg_arousal", "smile_rate",
    "avg_utterance_length", "agreement_word_count", "interrupt_rate", "avg_valence", "total_turns"
]
def apply_label_encodings(df, encoders, cols):
    for col in cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform([df[col].values[0]])
    return df

def run_inference(sensor_input: dict):
    df = pd.DataFrame([sensor_input])

    # --- MENTAL STATE ---
    df_mental = df[MENTAL_FEATURES]
    mental = {
        "stress_level": int(stress_model.predict(df_mental)[0]),
        "anxiety_level": str(anxiety_model.predict(df_mental)[0]),
        "motivation_score": int(motivation_model.predict(df_mental)[0])
    }

    # --- PHYSICAL STATE ---
 # Step 1: Apply label encodings
    df = apply_label_encodings(df, physical_package["label_encoders"],
                            ['workout_intensity', 'time_of_day', 'energy_level', 'fatigue_level', 'posture_label'])

    # Step 2: Extract only physical model features (after encoding)
    df_physical = df[physical_package["feature_columns"]]

    # Step 3: Predict using trained classifier and regressor
    Y_class_pred = physical_package["classifier"].predict(df_physical)
    Y_reg_pred = physical_package["regressor"].predict(df_physical)

    # Step 4: Zip results into output dict
    physical = dict(zip(
        physical_package["classification_targets"] + physical_package["regression_targets"],
        list(Y_class_pred[0]) + list(Y_reg_pred[0])
    ))

    # --- COGNITIVE STATE ---
    df_cog = df[cognitive_features]  # use the loaded feature list
    cog_pred = cognitive_model.predict(df_cog).iloc[0]  # returns a DataFrame

    cognitive = {
        "focus_level": cog_pred["focus_level"],
        "decision_making": cog_pred["decision_making"],
        "reaction_time": int(cog_pred["reaction_time"]),
        "accuracy": float(cog_pred["accuracy"])
    }

    # --- BIOMECHANICAL STATE ---
    df_bio = df[BIOMECHANICAL_FEATURES]
    biomechanical = {
        "balance_score": float(balance_model.predict(df_bio)[0]),
        "dexterity_score": float(dexterity_model.predict(df_bio)[0]),
        "posture_analysis": str(posture_model.predict(df_bio)[0])
    }

    # --- INTERPERSONAL STATE ---
    df_inter = df[INTERPERSONAL_FEATURES]
    interpersonal = {
        "leadership_score": float(leader_model.predict(df_inter)[0]),
        "social_engagement": engage_encoder.inverse_transform(engage_model.predict(df_inter))[0],
        "cooperation_score": float(coop_model.predict(df_inter)[0])
    }

    return {
        "mental": mental,
        "physical": physical,
        "cognitive": cognitive,
        "biomechanical": biomechanical,
        "interpersonal": interpersonal
    }
