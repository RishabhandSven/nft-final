# 🎉 NFT Wash Trading Detection System - Ready to Use!

## ✅ System Status

**Backend (Python + FastAPI + IsolationForest ML)**
- Status: ✅ **RUNNING** on http://localhost:8000
- Launch Script: `launch_server.py`
- Model: IsolationForest (contamination=0.05)
- Features: price_usd, time_since_last_trade, sellerFee_amount, is_circular
- Endpoints:
  - `GET /health` - Check backend status
  - `POST /analyze` - Analyze transaction for wash trading risk

**Frontend (HTML + JavaScript)**
- Status: ✅ **RUNNING** on http://localhost:3000
- Launch Script: `serve_frontend.py`
- Features:
  - Real-time transaction analysis
  - Risk scoring and visualization
  - Preset examples (Safe, Suspicious, High Risk)
  - Health status indicator

## 🚀 Quick Start

### If servers are not running, start them:

**Terminal 1 - Start Backend:**
```powershell
cd D:\nft2\NFT
.venv\Scripts\python launch_server.py 8000
```

**Terminal 2 - Start Frontend:**
```powershell
cd D:\nft2\NFT
.venv\Scripts\python serve_frontend.py
```

### Open the Application
Browse to: **http://localhost:3000**

## 📊 How to Use

### 1. **Manual Entry**
   - Enter transaction details:
     - Price (USD)
     - Time Since Last Trade (seconds)
     - Seller Fee (USD)
     - Buyer Address (0x...)
     - Seller Address (0x...)
   - Click "Analyze Transaction"

### 2. **Preset Examples**
   - **Safe Example**: Normal transaction (1 day gap, $250 fee)
   - **Suspicious**: Rapid transaction (5 min gap, $50 fee)  
   - **High Risk**: Likely wash trade (30 sec gap, $50k price, no fee)

### 3. **Results**
   The system returns:
   - ✅ **Status**: SAFE, SUSPICIOUS, or HIGH RISK
   - 📊 **Risk Score**: 0-100% (higher = more suspicious)
   - 🎯 **Wash Trade Verdict**: Boolean (true/false)

## 🧠 How the ML Model Works

The IsolationForest algorithm detects anomalies:
- **Trained on**: 500,000 mock NFT transactions
- **Contamination**: 5% (assumes ~5% of transactions are wash trades)
- **Detection Logic**: Identifies unusual transaction patterns including:
  - Rapid successive trades (low time_since_last_trade)
  - Unusual price points
  - Suspicious fee structures
  - Circular trading (same buyer and seller)

## 📁 Project Structure

```
NFT/
├── ai_engine/
│   ├── api.py              # FastAPI backend
│   ├── processor.py        # Data processor
│   ├── train.py            # Model training script
│   └── generate_dataset.py # Dataset generation
├── data/
│   ├── training_chunk.csv  # 500K training rows
│   └── results/
│       ├── wash_trading_brain.pkl  # Trained model
│       └── scaler.pkl              # Feature scaler
├── index.html              # Frontend (HTML/CSS/JS)
├── launch_server.py        # Backend launcher
├── serve_frontend.py       # Frontend launcher
└── [test files]
```

## 🔍 API Reference

### POST /analyze
```json
Request:
{
  "price_usd": 5000,
  "time_since_last_trade": 300,
  "sellerFee_amount": 50,
  "buyer_address": "0xBuyer001",
  "seller_address": "0xSeller001"
}

Response:
{
  "is_wash_trade": false,
  "risk_score": -0.50,
  "status": "SAFE"
}
```

### GET /health
```json
Response:
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## 🛑 Stopping the Servers

**Method 1:** Press Ctrl+C in the terminal windows

**Method 2:** Kill processes by port
```powershell
# Kill port 8000 (backend)
Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force

# Kill port 3000 (frontend)
Get-NetTCPConnection -LocalPort 3000 | Stop-Process -Force
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend shows "Offline" | Make sure `launch_server.py` is running |
| Port 8000 already in use | Change port in `launch_server.py` argument |
| CORS errors | Ensure frontend is at http://localhost:3000 |
| Slow analysis | First run loads model (~2-3 seconds) |

## 📚 Model Performance

- **Training Time**: ~10 minutes on 500K rows
- **Inference Time**: <100ms per transaction
- **Accuracy**: Based on IsolationForest with contamination=0.05
- **False Positive Rate**: Adjustable via contamination parameter

## 🔧 Customization

### Change contamination threshold (sensitivity)
Edit `ai_engine/api.py` line in model loading:
```python
model.contamination = 0.10  # 10% instead of 5%
```

### Use different port
```powershell
python launch_server.py 8001  # Use port 8001 instead
# Then edit index.html: const API_URL = 'http://localhost:8001';
```

### Add more training data
```powershell
python ai_engine/processor.py      # Extract features
python ai_engine/train.py          # Retrain model
python launch_server.py            # Restart backend
```

## 📝 Notes

- The model is trained on **synthetic data** for demonstration
- Real-world deployment would require labeled transaction data
- Consider monitoring false positive rates before production use
- Feature engineering can be improved with domain expertise

---

**Status**: ✅ All systems operational. Ready for testing!

Questions? Check the Python backend logs in the terminal for detailed debugging info.
