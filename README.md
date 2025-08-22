# Milk Packet Detector for Raspberry Pi

A minimal real-time milk packet detection and counting system for Raspberry Pi 4 with PiCamera.

## Features

- Real-time milk packet detection using TFLite model
- Automatic counting when packets cross a designated line
- Configurable FPS, resolution, and confidence thresholds
- Minimal design with clean white background and shadows
- Optimized for Raspberry Pi performance

## Hardware Requirements

- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- PiCamera v2 or v3
- MicroSD card (32GB recommended)

## Quick Start

### 1. Install Dependencies

```bash
chmod +x install.sh
./install.sh
```

### 2. Configure Settings

Edit `config.yaml` to adjust:
- Camera resolution and FPS
- Detection confidence threshold
- Model input dimensions
- Counting line position

### 3. Run Detector

```bash
source venv/bin/activate
python3 milk_detector.py
```

Press 'q' to quit the detection.

## Configuration

The `config.yaml` file controls all parameters:

- **FPS**: Target frames per second (lower = faster processing)
- **Resolution**: Camera frame size (lower = faster processing)
- **Confidence**: Detection threshold (0.0 to 1.0)
- **Counting Line**: Position where packets are counted

## How It Works

1. **Detection**: TFLite model processes each frame to detect milk packets
2. **Tracking**: Detected packets are tracked across frames
3. **Counting**: When a packet center crosses the counting line, count increments
4. **Display**: Real-time video with detection boxes, count, and FPS

## Performance Tips

- Lower resolution for faster processing
- Reduce FPS if needed for real-time performance
- Adjust confidence threshold based on lighting conditions
- Use SSD or fast microSD for better performance

## Troubleshooting

- **Camera not detected**: Ensure PiCamera is properly connected
- **Low FPS**: Reduce resolution or FPS in config
- **Poor detection**: Adjust confidence threshold and camera settings
- **Memory issues**: Close other applications, reduce resolution

## File Structure

```
├── milk_detector.py    # Main detection script
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
├── install.sh         # Installation script
├── model/             # TFLite model directory
│   └── best_float32.tflite
└── README.md          # This file
```

## License

This project is provided as-is for educational and commercial use. 