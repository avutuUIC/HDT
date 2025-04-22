from model_inference import run_inference
import openai
from pymongo import MongoClient

def format_state_prompt(state, profile):
    prompt = f"""
You are a digital twin interpreter. Based on the following information about the person and multimodal output data, generate a concise human-readable summary of the person's current physiological, psychological, biomechanical, cognitive, and interpersonal state.
Profile:
{profile}

MENTAL STATE: {state['mental']}
PHYSICAL STATE: {state['physical']}
COGNITIVE STATE: {state['cognitive']}
BIOMECHANICAL STATE: {state['biomechanical']}
INTERPERSONAL STATE: {state['interpersonal']}

Use professional and analytical tone. If something looks borderline, note it with uncertainty (e.g., "possibly elevated stress").

"""
    return prompt

def load_user_profile(user_id, db_name="digital_twin_db"):
    client = MongoClient("mongodb+srv://avutu:password%402001@cluster0.ox0gb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client[db_name]
    collection = db["user_profiles"]

    user_profile = collection.find_one({"user_id": user_id})
    client.close()

    if user_profile:
        return user_profile
    else:
        raise ValueError(f"No profile found for user_id {user_id}")


def get_sensor_inputs():
    # These are placeholder inputs simulating raw sensor/behavioral features for all models
    return {
        # Mental state inputs
        "eda": 0.65,
        "eda_delta": 0.03,
        "hrv_mean_hr": 72,
        "hrv_sdnn": 45,
        "hrv_rmssd": 30,
        "hrv_lf": 0.18,
        "hrv_hf": 0.12,
        "hrv_lf_hf": 1.5,

        # Physical state inputs
        "step_count": 6000,
        "heart_rate_avg": 78,
        "heart_rate_peak": 145,
        "skin_temp": 33.1,
        "respiration_rate": 17,
        "spo2_level": 98,
        "gps_distance_km": 5.2,
        "elevation_gain_m": 32,
        "deep_sleep_pct": 18,
        "rem_sleep_pct": 22,
        "light_sleep_pct": 60,
        "movement_during_sleep": 3.2,
        "resting_hr": 62,
        "hrv": 38,
        "oxygen_saturation": 98,
        "sleep_hours": 7,
        "workout_days": 3,
        "workout_intensity": "high",
        "calories_burned": 520,
        "time_since_rest": 20,
        "fatigue_score": 4,
        "perceived_soreness": 3,
        "perceived_exertion": 5,
        "temperature": 27.5,
        "humidity": 70,
        "time_of_day": "evening",
        "strength_score": 3.1,
        "endurance_score": 5.2,
        "energy_level": "normal",
        "fatigue_level": "medium",
        "recovery_index": 4.4,
        "injury_risk": 0.62,
        "accelerometer_x": 0.03,
        "accelerometer_y": 0.02,
        "accelerometer_z": 0.98,
        "gyroscope_x": 0.01,
        "gyroscope_y": 0.02,
        "gyroscope_z": 0.00,
        "accel_magnitude": 0.99,
        "gyro_magnitude": 0.022,
        "posture_label": "walking",

        # Cognitive state inputs
        "hr_mean": 72.5,
        "hr_std": 2.3,
        "gsr_mean": 1.05,
        "gsr_std": 0.1,
        "rr_mean": 0.36,
        "rr_std": 0.03,
        "temperature_mean": 33.4,
        "temperature_std": 0.12,
        "band_ax_mean": 0.08,
        "band_ax_std": 0.01,
        "band_ay_mean": 0.45,
        "band_ay_std": 0.03,
        "band_az_mean": 0.79,
        "band_az_std": 0.02,
        "time_on_task": 48000,
        "num_correct": 19,
        "num_incorrect": 1,

        # Biomechanical inputs
        "x_mean": 0.12,
        "x_std": 0.02,
        "x_min": -0.2,
        "x_max": 0.3,
        "x_range": 0.5,
        "x_median": 0.11,
        "x_iqr": 0.03,
        "y_mean": 0.08,
        "y_std": 0.01,
        "y_min": -0.1,
        "y_max": 0.25,
        "y_range": 0.35,
        "y_median": 0.07,
        "y_iqr": 0.025,
        "z_mean": 0.95,
        "z_std": 0.02,
        "z_min": 0.90,
        "z_max": 1.0,
        "z_range": 0.1,
        "z_median": 0.96,
        "z_iqr": 0.015,
        "magnitude_mean": 0.99,
        "magnitude_std": 0.01,
        "magnitude_min": 0.96,
        "magnitude_max": 1.02,
        "magnitude_range": 0.06,
        "magnitude_median": 0.985,
        "magnitude_iqr": 0.01,
        "jerk_mean": 0.02,
        "jerk_std": 0.005,
        "total_movement": 100,

        # Interpersonal state inputs
        "avg_power": 0.5,
        "total_speaking_time": 180,
        "initiations": 2,
        "avg_arousal": 0.7,
        "smile_rate": 0.8,
        "avg_utterance_length": 5,
        "agreement_word_count": 7,
        "interrupt_rate": 0.2,
        "avg_valence": 0.6,
        "total_turns": 30
    }

def call_openai(prompt, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable digital twin interpreter"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response['choices'][0]['message']['content']

def get_current_state(user_id):
    inputs = get_sensor_inputs()
    return run_inference(inputs)

def generate_state_summary(state, user_id, api_key):
    profile = load_user_profile(user_id)
    prompt = format_state_prompt(state, profile)
    return call_openai(prompt, api_key)



def main():
    user_id = "athlete_809"
    api_key = "..."

    if not api_key:
        raise EnvironmentError("Please set your OPENAI_API_KEY environment variable.")
    inputs = get_sensor_inputs()
    current_state = get_current_state(user_id)
    summary = generate_state_summary(current_state,user_id, api_key)
    print("\n--- Current Summary for", user_id, "---\n")
    print(summary)


if __name__ == "__main__":
    main()

