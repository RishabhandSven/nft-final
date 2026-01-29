# 🔍 NFT Wash Trading Detection System

A complete AI-powered system for detecting suspicious NFT transactions using Machine Learning. Features a Python FastAPI backend with an IsolationForest model and an interactive web frontend.

## ✨ Features

- **ML-Powered Detection**: IsolationForest algorithm trained on 500K transactions
- **Real-time Analysis**: Instant wash trading risk assessment
- **Interactive Dashboard**: Beautiful HTML/JS frontend with preset examples
- **REST API**: FastAPI backend with comprehensive endpoints
- **Risk Scoring**: Detailed anomaly detection with visualization

## 🏗️ System Architecture

```
NFT Wash Trading Detection
├── Backend (Python)
│   ├── FastAPI REST API (port 8000)
│   ├── IsolationForest ML Model
│   ├── StandardScaler for feature normalization
│   └── Endpoints: /health, /analyze
├── Frontend (HTML/JavaScript)
│   ├── HTTP Server (port 3000)
│   ├── Real-time analysis form
│   ├── Risk visualization
│   └── Preset examples (Safe, Suspicious, High Risk)
└── Data
    ├── 500K training transactions
    ├── Trained model (wash_trading_brain.pkl)
    └── Feature scaler (scaler.pkl)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip (Python package manager)

### 1. Setup Virtual Environment
```bash
cd NFT
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Start Backend Server
Open Terminal 1:
```bash
cd NFT
python launch_server.py 8000
```
Backend runs on: **http://localhost:8000**

### 3. Start Frontend Server
Open Terminal 2:
```bash
cd NFT
python serve_frontend.py
```
Frontend runs on: **http://localhost:3000**

### 4. Open Application
Navigate to: **http://localhost:3000**

## 📊 How to Use

### Manual Entry
1. Fill in transaction details:
   - **Price (USD)**: Transaction price
   - **Time Since Last Trade (seconds)**: Gap between trades
   - **Seller Fee (USD)**: Fee amount
   - **Buyer Address**: Buyer's wallet (0x...)
   - **Seller Address**: Seller's wallet (0x...)
2. Click "Analyze Transaction"
3. View risk assessment and score

### Preset Examples
- **Safe Example**: Normal transaction (1 day gap, $250 fee)
- **Suspicious**: Rapid transaction (5 min gap, $50 fee)
- **High Risk**: Likely wash trade (30 sec gap, $50k, no fee)

### Results
- ✅ **Status**: SAFE, SUSPICIOUS, or HIGH RISK
- 📊 **Risk Score**: 0-100% confidence
- 🎯 **Wash Trade Verdict**: Boolean result

## 🧠 Machine Learning Model

**Algorithm**: IsolationForest (Anomaly Detection)
- **Training Data**: 500,000 synthetic NFT transactions
- **Contamination**: 5% (assumes ~5% wash trades)
- **Features**:
  - `price_usd`: Transaction price
  - `time_since_last_trade`: Time gap in seconds
  - `sellerFee_amount`: Fee charged
  - `is_circular`: Same buyer/seller (0 or 1)

**Detection Logic**:
- Identifies unusual transaction patterns
- Flags rapid successive trades
- Detects suspicious fee structures
- Catches circular trading (same addresses)

## 📁 Project Structure

```
NFT/
├── ai_engine/
│   ├── api.py                      # FastAPI backend
│   ├── train.py                    # Model training
│   ├── processor.py                # Feature engineering
│   ├── generate_dataset.py         # Data generation
│   └── __pycache__/
├── data/
│   ├── training_chunk.csv          # 500K training rows
│   └── results/
│       ├── wash_trading_brain.pkl  # Trained model
│       ├── scaler.pkl              # Feature scaler
│       └── [ML metrics]
├── index.html                      # Frontend (HTML/CSS/JS)
├── launch_server.py                # Backend launcher
├── serve_frontend.py               # Frontend launcher
├── test_integration.py             # Integration tests
├── requirements.txt                # Python dependencies
├── SYSTEM_READY.md                 # Detailed docs
└── README.md                       # This file
```

## 🔌 API Reference

### GET /health
Check backend and model status
```json
Response: {
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### POST /analyze
Analyze transaction for wash trading risk
```json
Request: {
  "price_usd": 5000,
  "time_since_last_trade": 300,
  "sellerFee_amount": 50,
  "buyer_address": "0xBuyer001",
  "seller_address": "0xSeller001"
}

Response: {
  "is_wash_trade": false,
  "risk_score": -0.50,
  "status": "SAFE"
}
```

## 📦 Dependencies

```
fastapi==0.109.0
uvicorn==0.27.0
pandas==2.1.4
scikit-learn==1.3.2
joblib==1.3.2
requests==2.31.0
pydantic==2.5.2
```

Install with:
```bash
pip install -r requirements.txt
```

## 🛑 Stopping Servers

**Method 1**: Press `Ctrl+C` in terminal windows

**Method 2**: Kill by port (Windows)
```powershell
Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force  # Backend
Get-NetTCPConnection -LocalPort 3000 | Stop-Process -Force  # Frontend
```

**Method 2**: Kill by port (macOS/Linux)
```bash
lsof -ti:8000 | xargs kill -9   # Backend
lsof -ti:3000 | xargs kill -9   # Frontend
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend shows "Offline" | Ensure `launch_server.py` is running |
| Port already in use | Change port in server script or kill process |
| CORS errors | Verify frontend is at `http://localhost:3000` |
| Slow first request | Model loads on first request (~2-3 seconds) |
| Model not found error | Run `python ai_engine/train.py` to retrain |

## 🔧 Customization

### Change Sensitivity
Edit `ai_engine/api.py` to adjust contamination:
```python
model.contamination = 0.10  # 10% instead of 5%
```

### Use Different Port
```bash
python launch_server.py 8001  # Backend on 8001
# Then update API_URL in index.html
```

### Retrain Model
```bash
python ai_engine/processor.py      # Extract features
python ai_engine/train.py          # Retrain model
python launch_server.py            # Restart
```

## 📈 Model Performance

- **Training Time**: ~10 minutes on 500K rows
- **Inference Time**: <100ms per transaction
- **Memory**: ~2GB with training data loaded
- **Accuracy**: Based on contamination parameter

## ⚠️ Notes

- Model trained on **synthetic data** for demonstration
- Real production use requires labeled transaction data
- Contamination parameter affects false positive rate
- Consider domain expertise for feature engineering improvements

## 🎯 Next Steps

- [ ] Integrate with real NFT marketplace data
- [ ] Add historical transaction tracking
- [ ] Implement alert system for high-risk transactions
- [ ] Create admin dashboard with analytics
- [ ] Add blockchain verification
- [ ] Deploy to cloud platform (AWS/Azure/GCP)

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Created with ❤️ for NFT security research

---

**Status**: ✅ All systems operational and tested

For issues or questions, refer to SYSTEM_READY.md for detailed troubleshooting guide.

python scripts/results_analysis.py

## Contact

For access to the tagged datasets or ML samples, please contact Niuniu Zhang at [niuniu.zhang.phd@anderson.ucla.edu](mailto:niuniu.zhang.phd@anderson.ucla.edu) or [niuniu@ucla.edu](mailto:niuniu@ucla.edu).
