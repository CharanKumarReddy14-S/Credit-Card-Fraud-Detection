#!/bin/bash

# Credit Card Fraud Detection - Setup Script
# This script automates the setup and execution of the project

set -e  # Exit on error

echo "=========================================="
echo "Credit Card Fraud Detection - Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo "1. Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
echo ""
echo "4. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dependencies installed"

# Create necessary directories
echo ""
echo "5. Creating project directories..."
mkdir -p data/processed
mkdir -p models
mkdir -p reports
mkdir -p uploads
print_success "Directories created"

# Check for dataset
echo ""
echo "6. Checking for dataset..."
if [ -f "data/creditcard.csv" ]; then
    print_success "Dataset found"
    
    # Ask user if they want to train models
    echo ""
    read -p "Dataset found. Do you want to train models now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "7. Training models (this may take 10-30 minutes)..."
        python train_model.py
        print_success "Model training completed"
        
        # Generate evaluation reports
        echo ""
        echo "8. Generating evaluation reports..."
        python evaluate_model.py
        print_success "Evaluation reports generated"
    else
        print_info "Skipping model training. Run 'python train_model.py' when ready."
    fi
else
    print_error "Dataset not found at data/creditcard.csv"
    echo ""
    print_info "Please download the dataset from:"
    print_info "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo ""
    print_info "Place 'creditcard.csv' in the 'data/' directory and run this script again."
fi

# Setup complete
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the API server:"
echo "   python -m uvicorn api.main:app --reload"
echo ""
echo "2. In a new terminal, start the Streamlit UI:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "3. Or use Docker:"
echo "   docker-compose up --build"
echo ""
echo "API will be available at: http://localhost:8000"
echo "UI will be available at: http://localhost:8501"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo ""
print_success "Happy fraud detecting! 💳"