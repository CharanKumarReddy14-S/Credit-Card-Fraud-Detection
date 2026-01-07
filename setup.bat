@echo off
REM Credit Card Fraud Detection - Windows Setup Script

echo ==========================================
echo Credit Card Fraud Detection - Setup
echo ==========================================
echo.

REM Check Python
echo 1. Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher.
    pause
    exit /b 1
)
echo [SUCCESS] Python found
echo.

REM Create virtual environment
echo 2. Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 3. Activating virtual environment...
call venv\Scripts\activate.bat
echo [SUCCESS] Virtual environment activated
echo.

REM Install dependencies
echo 4. Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed
echo.

REM Create directories
echo 5. Creating project directories...
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "reports" mkdir reports
if not exist "uploads" mkdir uploads
echo [SUCCESS] Directories created
echo.

REM Check for dataset
echo 6. Checking for dataset...
if exist "data\creditcard.csv" (
    echo [SUCCESS] Dataset found
    echo.
    set /p TRAIN="Dataset found. Do you want to train models now? (y/n): "
    if /i "%TRAIN%"=="y" (
        echo.
        echo 7. Training models (this may take 10-30 minutes)...
        python train_model.py
        if %errorlevel% equ 0 (
            echo [SUCCESS] Model training completed
            echo.
            echo 8. Generating evaluation reports...
            python evaluate_model.py
            echo [SUCCESS] Evaluation reports generated
        ) else (
            echo [ERROR] Model training failed
        )
    ) else (
        echo [INFO] Skipping model training. Run 'python train_model.py' when ready.
    )
) else (
    echo [ERROR] Dataset not found at data\creditcard.csv
    echo.
    echo [INFO] Please download the dataset from:
    echo https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    echo.
    echo Place 'creditcard.csv' in the 'data\' directory and run this script again.
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Start the API server:
echo    python -m uvicorn api.main:app --reload
echo.
echo 2. In a new terminal, start the Streamlit UI:
echo    streamlit run streamlit_app.py
echo.
echo 3. Or use Docker:
echo    docker-compose up --build
echo.
echo API will be available at: http://localhost:8000
echo UI will be available at: http://localhost:8501
echo.
echo API Documentation: http://localhost:8000/docs
echo.
echo [SUCCESS] Happy fraud detecting! 💳
echo.
pause