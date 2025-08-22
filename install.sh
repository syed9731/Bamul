#!/bin/bash

echo "Installing Milk Packet Detector on Raspberry Pi..."

# # Update system
# echo "Updating system packages..."
# sudo apt update && sudo apt upgrade -y

# # Install system dependencies
# echo "Installing system dependencies..."
# sudo apt install -y python3-pip python3-venv python3-opencv
# sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
# sudo apt install -y libjasper-dev libqtcore4 libqtgui4 libqt4-test

# # Install PiCamera2
# echo "Installing PiCamera2..."
# sudo apt install -y python3-picamera2

# # Create virtual environment
# echo "Creating Python virtual environment..."
# python3 -m venv venv
# source venv/bin/activate

# # Upgrade pip
# pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installation complete!"
echo ""
echo "To run the detector:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python3 milk_detector.py"
echo ""
echo "Make sure your TFLite model is in the model/ directory" 