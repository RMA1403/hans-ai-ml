from flask import Flask, request, jsonify
import calorie_intake.calorie_intake as ci
from recipes.rag import rag_model
from recipes.prompt import generate_prompt

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World'

@app.route('/recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()

        calorie = data.get("calories")
        ingredients = data.get("ingredients")

        prompt = generate_prompt(calorie, ingredients)

        response = rag_model.generate_content(prompt)

        return jsonify({
            "recipe": str(response.text).strip()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/calorie-intake', methods=['POST'])
def predict_calorie_intake():
    try:
        data = request.get_json()

        result = ci.full_prediction_pipeline(
            age=int(data.get("age")),
            gender=data.get("gender"),
            daily_calories_consumed=float(data.get("dailyCaloriesConsumed")),
            weight_change_in_lbs=float(data.get("weightChangeInLbs")),
            duration_in_weeks=float(data.get("durationInWeeks")),
            physical_activity_level=data.get("physicalActivityLevel"),
            sleep_quality=data.get("sleepQuality"),
            stress_level=int(data.get("stressLevel")),
            current_weight_in_lbs=float(data.get("currentWeightInLbs")),
            caloric_adjustment=float(data.get("caloricAdjustment"))
        )

        return jsonify({
            "prediction": result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
