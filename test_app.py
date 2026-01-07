"""
Unit tests for FastAPI backend
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "online"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction data"""
        return {
            "Time": 0,
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            "V5": -0.338321,
            "V6": 0.462388,
            "V7": 0.239599,
            "V8": 0.098698,
            "V9": 0.363787,
            "V10": 0.090794,
            "V11": -0.551600,
            "V12": -0.617801,
            "V13": -0.991390,
            "V14": -0.311169,
            "V15": 1.468177,
            "V16": -0.470401,
            "V17": 0.207971,
            "V18": 0.025791,
            "V19": 0.403993,
            "V20": 0.251412,
            "V21": -0.018307,
            "V22": 0.277838,
            "V23": -0.110474,
            "V24": 0.066928,
            "V25": 0.128539,
            "V26": -0.189115,
            "V27": 0.133558,
            "V28": -0.021053,
            "Amount": 149.62
        }
    
    def test_predict_endpoint(self, sample_transaction):
        """Test single prediction endpoint"""
        response = client.post("/predict", json=sample_transaction)
        
        # Check status code
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "risk_level" in data
            assert "timestamp" in data
            assert isinstance(data["is_fraud"], bool)
            assert 0 <= data["fraud_probability"] <= 1
    
    def test_predict_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {"Time": 0, "Amount": 100}  # Missing features
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "features_count" in data


class TestValidation:
    """Test input validation"""
    
    def test_missing_required_fields(self):
        """Test with missing required fields"""
        incomplete_data = {
            "Time": 0,
            "Amount": 100
            # Missing V1-V28
        }
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test with invalid data types"""
        invalid_data = {
            "Time": "invalid",  # Should be float
            "V1": -1.359807,
            # ... other fields
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling"""
    
    def test_upload_non_csv_file(self):
        """Test uploading non-CSV file"""
        # This would require actual file upload simulation
        pass
    
    def test_upload_empty_file(self):
        """Test uploading empty file"""
        # This would require actual file upload simulation
        pass


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])