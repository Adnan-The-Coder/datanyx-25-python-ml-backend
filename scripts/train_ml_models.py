"""Train ML models using the compact primary dataset to predict disease presence.

This script reads the primary_dataset.csv (compact format with 7 disease columns)
and trains individual binary classifiers for each disease. Since the current dataset
only contains disease presence flags (0/1), we'll simulate feature data or use
cross-validation within the existing data.
"""
import os
import csv
import pickle
import random
import sys

# Add parent directory to path to import our shared module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ml_models import SimpleRandomForest, load_primary_dataset, generate_synthetic_features

# Simple ML implementation without sklearn dependency
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_names = []
        random.seed(random_state)
        np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        n = len(X)
        indices = [random.randint(0, n-1) for _ in range(n)]
        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]
        return X_boot, y_boot
    
    def _best_split(self, X, y, max_features=None):
        if max_features is None:
            max_features = int(np.sqrt(len(X[0]))) + 1
        
        n_features = len(X[0])
        features_to_try = random.sample(range(n_features), min(max_features, n_features))
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in features_to_try:
            values = [row[feature_idx] for row in X]
            unique_values = list(set(values))
            
            for threshold in unique_values:
                left_y = [y[i] for i in range(len(y)) if X[i][feature_idx] <= threshold]
                right_y = [y[i] for i in range(len(y)) if X[i][feature_idx] > threshold]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                total = len(y)
                left_gini = self._gini_impurity(left_y)
                right_gini = self._gini_impurity(right_y)
                weighted_gini = (len(left_y) / total) * left_gini + (len(right_y) / total) * right_gini
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        classes = list(set(y))
        gini = 1.0
        for cls in classes:
            prob = y.count(cls) / len(y)
            gini -= prob ** 2
        return gini
    
    def _build_tree(self, X, y, depth=0):
        # Base cases
        if depth >= self.max_depth or len(set(y)) <= 1 or len(y) < 2:
            # Return majority class
            if len(y) == 0:
                return 0
            return max(set(y), key=y.count)
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return max(set(y), key=y.count)
        
        # Split data
        left_X = [X[i] for i in range(len(X)) if X[i][feature] <= threshold]
        left_y = [y[i] for i in range(len(y)) if X[i][feature] <= threshold]
        right_X = [X[i] for i in range(len(X)) if X[i][feature] > threshold]
        right_y = [y[i] for i in range(len(y)) if X[i][feature] > threshold]
        
        # Build subtrees
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree = self._build_tree(X_boot, y_boot)
            self.trees.append(tree)
        return self
    
    def _predict_tree(self, tree, sample):
        if not isinstance(tree, dict):
            return tree  # Leaf node
        
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], sample)
        else:
            return self._predict_tree(tree['right'], sample)
    
    def predict(self, X):
        predictions = []
        for sample in X:
            votes = [self._predict_tree(tree, sample) for tree in self.trees]
            # Majority vote
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)
        return predictions


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']


def load_data():
    """Load the compact primary dataset."""
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def generate_synthetic_features(rows):
    """Generate synthetic features for training since we only have labels.
    
    In a real scenario, you would use the original 70+ feature columns.
    Here we create plausible synthetic features correlated with the diseases.
    """
    features = []
    labels = {disease: [] for disease in DISEASES}
    
    for row in rows:
        # Generate synthetic features (age, severity scores, etc.)
        # These would normally come from your original feature dataset
        age = random.uniform(20, 80)
        bmi = random.uniform(18, 35)
        
        # Disease-specific synthetic features
        diplopia_severity = random.uniform(0, 10) if int(row['diplopia']) else random.uniform(0, 3)
        bulbar_score = random.uniform(0, 10) if int(row['bulbar']) else random.uniform(0, 2)
        facial_weakness = random.uniform(0, 10) if int(row['facial']) else random.uniform(0, 2)
        fatigue_level = random.uniform(0, 10) if int(row['fatigue']) else random.uniform(0, 3)
        limb_strength = random.uniform(0, 10) if int(row['limb']) else random.uniform(7, 10)
        ptosis_degree = random.uniform(0, 5) if int(row['ptosis']) else random.uniform(0, 1)
        resp_function = random.uniform(30, 80) if int(row['respiratory']) else random.uniform(80, 100)
        
        feature_vector = [
            age, bmi, diplopia_severity, bulbar_score, facial_weakness,
            fatigue_level, limb_strength, ptosis_degree, resp_function
        ]
        features.append(feature_vector)
        
        # Extract labels
        for disease in DISEASES:
            labels[disease].append(int(row[disease]))
    
    return features, labels


def train_models():
    """Train individual models for each disease."""
    rows = load_data()
    X, y_dict = generate_synthetic_features(rows)
    
    feature_names = [
        'age', 'bmi', 'diplopia_severity', 'bulbar_score', 'facial_weakness',
        'fatigue_level', 'limb_strength', 'ptosis_degree', 'resp_function'
    ]
    
    models = {}
    accuracies = {}
    
    for disease in DISEASES:
        print(f"Training model for {disease}...")
        
        y = y_dict[disease]
        
        # Simple train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = SimpleRandomForest(n_trees=20, max_depth=6, random_state=42)
        model.feature_names = feature_names
        model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = model.predict(X_test)
        accuracy = sum(1 for true, pred in zip(y_test, y_pred) if true == pred) / len(y_test)
        accuracies[disease] = accuracy
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': feature_names,
                'accuracy': accuracy
            }, f)
        
        models[disease] = model
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Saved to: {model_path}")
    
    return models, accuracies


def predict_single_sample(feature_vector):
    """Make predictions for a single patient using trained models."""
    predictions = {}
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                obj = pickle.load(f)
            model = obj['model']
            pred = model.predict([feature_vector])[0]
            predictions[disease] = int(pred)
        except Exception as e:
            print(f"Error loading model for {disease}: {e}")
            predictions[disease] = 0
    
    return predictions


if __name__ == '__main__':
    print("Training ML models for disease prediction...")
    models, accuracies = train_models()
    
    print("\nTraining complete!")
    print("Model accuracies:")
    for disease, acc in accuracies.items():
        print(f"  {disease}: {acc:.3f}")
    
    # Test prediction
    print("\nTesting prediction on sample data...")
    test_features = [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]  # Sample features
    predictions = predict_single_sample(test_features)
    print("Sample predictions:", predictions)