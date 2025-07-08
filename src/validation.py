import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

class ValidationEngine:
    def __init__(self):
        self.anomaly_model = self._train_anomaly_model()
        
    def _train_anomaly_model(self, n_samples=1000):
        """Train fraud detection model"""
        # Generate normal transactions (85%)
        normal_totals = np.random.normal(500, 200, int(n_samples * 0.85))
        
        # Generate anomalies (15%)
        anomalies = np.concatenate([
            np.random.uniform(-1000, -1, int(n_samples * 0.05)),
            np.random.uniform(3000, 10000, int(n_samples * 0.10))
        ])
        
        X = np.concatenate([normal_totals, anomalies]).reshape(-1, 1)
        model = IsolationForest(contamination=0.15, random_state=42)
        model.fit(X)
        return model
    
    def validate(self, data):
        """Validate extracted data"""
        errors = []
        
        # Date validation
        if 'date' in data:
            try:
                invoice_date = datetime.strptime(data['date'], '%Y-%m-%d')
                if invoice_date > datetime.now():
                    errors.append("Future date detected")
            except:
                errors.append("Invalid date format")
        
        # Total validation
        if 'total' in data:
            try:
                total = float(data['total'].replace(',', ''))
                
                # Amount validation
                if total <= 0:
                    errors.append("Invalid total amount")
                
                # Line item validation
                if 'items' in data:
                    calculated = sum(
                        float(item['price']) * float(item['quantity']) 
                        for item in data['items']
                    )
                    if abs(total - calculated) > 0.01:
                        errors.append(f"Total mismatch: {total} vs {calculated}")
                
                # Fraud detection
                if self.anomaly_model.predict([[total]])[0] == -1:
                    errors.append("Suspicious transaction amount")
                    
            except ValueError:
                errors.append("Invalid total format")
                
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }# Fraud detection logic
