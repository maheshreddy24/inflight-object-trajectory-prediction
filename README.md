# Inflight Object Trajectory Prediction

## **Overview**
Inflight Object Trajectory Prediction is a deep learning-based framework that predicts the trajectory of objects in motion. The model utilizes **vision-based** and **motion capture-based** data to improve accuracy.  

- **Vision-Based Approach:** Uses **3D CNNs** for spatiotemporal feature extraction.  
- **Motion Capture-Based Approach:** Includes **LSTM and Transformer models** for sequence-based trajectory forecasting.  

---

## **Codebase Structure**
```
Inflight-Object-Trajectory-Prediction/
│── experiments/        # Contains pretrained weights and experiment logs
│── vision/             # Vision-based trajectory prediction (3D CNNs)
│   ├── datasets.py     # Data loading and preprocessing functions
│   ├── main.py         # Entry point for training and evaluation
│   ├── models.py       # Model architecture (3D CNNs for trajectory prediction)
│   ├── optimisation.py # Optimization algorithms and loss functions
│── motion_capture/     # Motion capture-based prediction (LSTMs & Transformers)
│   ├── lstm.ipynb      # LSTM model for trajectory prediction
│   ├── transformer.ipynb  # Transformer-based model
│── requirements.txt    # Dependency list
│── README.md           # Documentation
```

---

## **Installation**
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/maheshreddy24/inflight-object-trajectory-prediction.git
cd inflight-object-trajectory-prediction
pip install -r requirements.txt
```

---

## **Usage**

### **Running Vision-Based Prediction (3D CNN)**
To train or evaluate the vision-based model, run:

```bash
python vision/main.py
```
This script handles **data loading, model training, and evaluation**.

### **Running Motion Capture Prediction**

#### **LSTM-Based Model**
```bash
jupyter notebook motion_capture/lstm.ipynb
```

#### **Transformer-Based Model**
```bash
jupyter notebook motion_capture/transformer.ipynb
```

---

## **Pretrained Weights**
Pretrained models are available in the `experiments/` directory. To use them for evaluation, modify the configuration file and load the corresponding checkpoint.

---

## **Future Work**
- Enhancing **3D CNN-based vision models** for better spatiotemporal feature extraction.
- Improving **motion capture-based Transformer** architectures for better accuracy.
- Extending support for **more diverse objects and complex motion types**.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

