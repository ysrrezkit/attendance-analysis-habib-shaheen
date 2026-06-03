@echo off
echo Checking for required libraries...
pip install -r requirements.txt

echo Starting the dashboard...
python app.py

pause