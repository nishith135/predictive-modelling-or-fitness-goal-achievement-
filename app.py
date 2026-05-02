import pandas as pd
import joblib
import re
import sqlite3
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================
workout_model = joblib.load("Models/Voting_Workout_model.sav")
nutrition_model = joblib.load("Models/Voting_Nutrition_model.sav")
label_encoders = joblib.load("Models/label_encoders.pkl")

# Model feature order (must match training)
workout_feature_columns = [str(col) for col in workout_model.feature_names_in_]
nutrition_feature_columns = [str(col) for col in nutrition_model.feature_names_in_]

# Categorical features that need LabelEncoder transform
categorical_feature_columns = [
    col for col in label_encoders.keys() if col in workout_feature_columns
]

# Binary numeric columns coming from dropdowns (0/1)
binary_numeric_columns = {"has_allergy", "has_digestive_issue", "smoking", "alcohol"}


# ==============================
# HOME PAGE
# ==============================

@app.route('/home')
def home():
    return render_template("home.html")


# ==============================
# PREDICTION
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}

    try:
        form_payload = request.form.to_dict(flat=True)
        if not form_payload:
            return render_template("home.html", error="No form data received. Please submit from the home form.")

        # 1) Collect and cast form inputs for stage 1 model
        for col in workout_feature_columns:
            raw_val = request.form.get(col, "").strip()
            if raw_val == "":
                raise ValueError(f"Missing required field: {col}")

            if col in categorical_feature_columns:
                input_data[col] = raw_val
            elif col in binary_numeric_columns:
                input_data[col] = int(float(raw_val))
            else:
                input_data[col] = float(raw_val)
        
        # Debug: now input_data is populated and used for prediction
        print("Received fields:", list(form_payload.keys()))
        print("Parsed input_data:", input_data)

        # 2) Encode categorical features
        for col in categorical_feature_columns:
            try:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            except ValueError:
                raise ValueError(f"Invalid value for {col}: {input_data[col]}")

        # 3) Stage 1 input dataframe
        df_stage1 = pd.DataFrame([input_data], columns=workout_feature_columns)

        # 4) Stage 1 prediction
        workout_pred = int(workout_model.predict(df_stage1)[0])

        # 5) Stage 2 input (append predicted_workout_plan in exact expected order)
        df_stage2 = df_stage1.copy()
        df_stage2["predicted_workout_plan"] = workout_pred
        df_stage2 = df_stage2.reindex(columns=nutrition_feature_columns)

        # 6) Stage 2 prediction
        nutrition_pred = int(nutrition_model.predict(df_stage2)[0])

        # 7) Decode class labels
        workout_label = label_encoders['workout_plan_level'].inverse_transform([workout_pred])[0]
        nutrition_label = label_encoders['nutrition_plan_level'].inverse_transform([nutrition_pred])[0]

        return render_template(
            "result.html",
            workout_result=workout_label,
            nutrition_result=nutrition_label,
            **form_payload
        )

    except ValueError as err:
        return render_template("home.html", error=str(err))
    except Exception:
        return render_template("home.html", error="Unable to process prediction. Please verify your inputs.")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')



@app.route('/graphs')
def graphs():
	return render_template('graphs.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

# ==============================
if __name__ == "__main__":
    app.run(debug=True)
