"""Simple HTTP server to test ML models without complex dependencies."""
import http.server
import socketserver
import json
import pickle
import os
import sys
import urllib.parse as urlparse

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ml_models import SimpleRandomForest

PORT = 8000
MODELS_DIR = os.path.join(parent_dir, 'models')
DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']

class MLHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "message": "ML Prediction API is running!",
                "endpoints": {
                    "/": "API info",
                    "/health": "Health check",
                    "/models": "Available models info",
                    "/predict": "POST - Make predictions (send JSON with patient features)"
                }
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "models_loaded": len(DISEASES)}
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            models_info = {}
            for disease in DISEASES:
                model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                        models_info[disease] = {
                            "available": True,
                            "accuracy": model_data.get('accuracy', 'unknown')
                        }
                    except:
                        models_info[disease] = {"available": False, "error": "Failed to load"}
                else:
                    models_info[disease] = {"available": False, "error": "Model file not found"}
            
            response = {"models": models_info}
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())

    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                # Expect features in the format: [age, bmi, symptom_duration, ...]
                features = data.get('features', [])
                if len(features) != 9:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {"error": "Expected 9 features: [age, bmi, symptom_duration, severity, progression, medication_response, exercise_tolerance, stress_impact, health_score]"}
                    self.wfile.write(json.dumps(error_response).encode())
                    return
                
                # Make predictions for all diseases
                predictions = {}
                for disease in DISEASES:
                    model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
                    if os.path.exists(model_path):
                        try:
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                            model = model_data['model']
                            prediction = model.predict([features])[0]
                            predictions[disease] = {
                                "prediction": int(prediction),
                                "status": "Present" if prediction == 1 else "Absent",
                                "model_accuracy": model_data.get('accuracy', 'unknown')
                            }
                        except Exception as e:
                            predictions[disease] = {"error": str(e)}
                    else:
                        predictions[disease] = {"error": "Model not found"}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "patient_features": features,
                    "predictions": predictions
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"error": f"Internal server error: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MLHandler) as httpd:
        print(f"ML Prediction API server running at http://localhost:{PORT}")
        print("Available endpoints:")
        print("  GET  /          - API info")
        print("  GET  /health    - Health check")
        print("  GET  /models    - Models information")
        print("  POST /predict   - Make predictions")
        print("\nExample prediction request:")
        print("POST /predict")
        print('{"features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]}')
        print("\nPress Ctrl+C to stop the server")
        httpd.serve_forever()