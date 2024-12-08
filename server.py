from flask import Flask, request, jsonify
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import calorie_intake.calorie_intake as ci

PROJECT_ID = 'bangkit-capstone-hans-ai'
REGION = 'us-central1'
DISPLAY_NAME = "test_corpus"
PATHS = [
    "https://drive.google.com/file/d/1_KguMbb4AeeCPqJD16MAf6l8vUy1KhPL/view?usp=sharing"
]
MODEL_NAME = "gemini-1.5-flash-002"

embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-004"
)

rag_corpus = rag.create_corpus(
    display_name=DISPLAY_NAME,
    embedding_model_config=embedding_model_config
)

rag.import_files(
    rag_corpus.name,
    PATHS,
    chunk_size=512,
    chunk_overlap=100,
    max_embedding_requests_per_min=900,
)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.8
        )
    )
)

rag_model = GenerativeModel(
    model_name=MODEL_NAME,
    tools=[rag_retrieval_tool]
)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World'

@app.route('/generate')
def generate():
    prompt = request.args.get("prompt")
    response = rag_model.generate_content(prompt)

    return {
        "response": response.text
    }

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
