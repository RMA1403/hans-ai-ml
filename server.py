from flask import Flask, request
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool

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