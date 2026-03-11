from flask import Flask, request, jsonify
import joblib

# Chargement des modèles (décommente quand tu seras prêt, pour tester le démarrage d'abord)
# model = joblib.load('models/best_estimator_TfidfVectorizer_80000_LogisticRegression.joblib')
# vectorizer = joblib.load('models/vectorizer_TfidfVectorizer_100000.pkl')
# mlb = joblib.load('models/mlb_100000.pkl')

app = Flask(__name__)

# ~ def prediction(req_data: str) -> List[str]:

    # result = model.predict(vectorizer.transform([req_data]))
    # result = mlb.inverse_transform(result)[0]
    # return list(result)  # tuple → list pour JSON



@app.route("/", methods=["GET"])
def home():
    return "<p>Hello, World!</p>"


# ~ @app.route("/predict", methods=["POST"])
# ~ def predict():
    # ~ # Récupère le body brut (comme en FastAPI)
    # ~ req_data = request.get_data(as_text=True)
    
    # ~ if not req_data:
        # ~ return jsonify({"error": "Aucun texte fourni"}), 400
    
    # ~ try:
        # ~ tags = prediction(req_data)
        # ~ return jsonify({"tags": tags})
    # ~ except Exception as e:
        # ~ return jsonify({"error": str(e)}), 500


