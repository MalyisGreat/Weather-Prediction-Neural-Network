# Weather Prediction Neural Network

This repository contains an advanced time-series forecasting model that predicts weather conditions and provides the confidence of each forecast. The core of the system is a Temporal Convolutional Network (TCN) combined with a Mixture Density Network (MDN) to produce probabilistic predictions.

![Model Diagram](Figure_3.png)

## Features

- **Probabilistic forecasting** using a TCN + MDN architecture
- **Automatic checkpointing and resume** capability
- **Early stopping** based on validation metrics
- **Supports GPU acceleration** when available
- **Pre-trained models** included for quick experimentation

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

Install dependencies with pip:

```bash
pip install torch pandas numpy scikit-learn joblib matplotlib
```

## Training

Run `advanced.py` to train the model from scratch or resume from the last checkpoint:

```bash
python advanced.py
```

The script automatically downloads the Beijing PM2.5 dataset from the UCI repository and performs feature engineering before training.

## Repository Structure

- `advanced.py` – main training script
- `Figure_3.png` – model architecture diagram
- `weather_tcn_mdn_best.pth` – best-performing model weights
- `weather_tcn_mdn_full.pth` – final model after training
- `weather_scaler.pkl` / `weather_encoder.pkl` – preprocessing artifacts

## License

This project is released under the MIT License.
