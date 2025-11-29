import random
import csv
import os

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        n_samples = len(X)
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = [random.randint(0, n_samples-1) for _ in range(n_samples)]
            X_bootstrap = [X[i] for i in indices]
            y_bootstrap = [y[i] for i in indices]
            
            # Create and train a decision tree
            tree = SimpleDecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        """Make predictions using the random forest"""
        if not self.trees:
            return [0] * len(X)
        
        predictions = []
        for x in X:
            # Get predictions from all trees
            tree_predictions = [tree.predict_single(x) for tree in self.trees]
            # Majority vote
            prediction = 1 if sum(tree_predictions) > len(tree_predictions) / 2 else 0
            predictions.append(prediction)
        
        return predictions

class SimpleDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self._build_tree(X, y, 0)
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stop conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(set(y)) == 1):
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gini = self._find_best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # Split the data
        left_indices, right_indices = self._split_data(X, best_feature, best_threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return self._create_leaf(y)
        
        # Create child nodes
        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]
        
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }
    
    def _create_leaf(self, y):
        """Create a leaf node with the most common class"""
        if not y:
            return {'class': 0}
        
        counts = {0: 0, 1: 0}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        most_common_class = max(counts, key=counts.get)
        return {'class': most_common_class}
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0]) if X else 0
        
        # Try random subset of features (for random forest)
        features_to_try = list(range(n_features))
        if n_features > 1:
            n_features_to_try = max(1, int(n_features ** 0.5))
            features_to_try = random.sample(range(n_features), n_features_to_try)
        
        for feature_idx in features_to_try:
            # Get all feature values and sort them
            feature_values = sorted(set(row[feature_idx] for row in X))
            
            # Try splitting at each unique value
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                left_indices, right_indices = self._split_data(X, feature_idx, threshold)
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]
                
                n_total = len(y)
                n_left = len(y_left)
                n_right = len(y_right)
                
                gini_left = self._calculate_gini(y_left)
                gini_right = self._calculate_gini(y_right)
                
                weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def _split_data(self, X, feature_idx, threshold):
        """Split data based on feature and threshold"""
        left_indices = []
        right_indices = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices
    
    def _calculate_gini(self, y):
        """Calculate Gini impurity"""
        if not y:
            return 0
        
        n_samples = len(y)
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        gini = 1.0
        for count in counts.values():
            prob = count / n_samples
            gini -= prob ** 2
        
        return gini
    
    def predict_single(self, x):
        """Make prediction for a single sample"""
        if self.tree is None:
            return 0
        
        return self._traverse_tree(x, self.tree)
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction"""
        if 'class' in node:
            return node['class']
        
        feature_value = x[node['feature']]
        if feature_value <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

def load_primary_dataset():
    """Load the primary dataset"""
    data = []
    
    # Try different possible paths for the dataset
    possible_paths = [
        'primary_dataset.csv',
        'data/primary_dataset.csv',
        os.path.join('data', 'primary_dataset.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'primary_dataset.csv')
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data.append(row)
                print(f"Successfully loaded dataset from: {path}")
                return data
        except FileNotFoundError:
            continue
    
    print("Primary dataset not found in any of the expected locations!")
    print(f"Tried paths: {possible_paths}")
    return []

def generate_synthetic_features(primary_data):
    """Generate synthetic features for ML training"""
    features = []
    labels = {disease: [] for disease in ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']}
    
    for patient in primary_data:
        # Generate synthetic features (age, bmi, symptom_duration, etc.)
        base_features = [
            random.uniform(20, 80),  # age
            random.uniform(18, 35),  # bmi
            random.uniform(0.1, 10), # symptom_duration_years
            random.uniform(0.1, 10), # symptom_severity_scale
            random.uniform(0.1, 10), # weakness_progression_rate
            random.uniform(1, 10),   # medication_response_scale
            random.uniform(1, 10),   # exercise_tolerance
            random.uniform(0.1, 5),  # stress_impact_scale
            random.uniform(50, 100)  # overall_health_score
        ]
        
        features.append(base_features)
        
        # Add labels for each disease
        for disease in labels.keys():
            labels[disease].append(int(patient[disease]))
    
    return features, labels