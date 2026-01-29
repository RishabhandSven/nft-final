# React Frontend Setup for Wash Trading Detector

## Step 1: Keep the Backend Running

The Python FastAPI server must continue running on `http://127.0.0.1:8000` while developing the frontend.

**To start the backend (if not already running):**
```powershell
cd D:\nft2\NFT
.venv\Scripts\python -m uvicorn ai_engine.api:app --port 8000
```

The server should print:
```
[*] Training Model on Startup...
[*] Looking for brain at: D:\nft2\NFT\data\results\wash_trading_brain.pkl
[*] Model exists? True
[*] Scaler exists? True
[OK] Brain Loaded Successfully
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Step 2: Set Up Your React Project

If you don't have a React project yet, create one:
```bash
npx create-react-app nft-wash-trader
cd nft-wash-trader
```

Or with Vite:
```bash
npm create vite@latest nft-wash-trader -- --template react
cd nft-wash-trader
npm install
```

## Step 3: Add the API Service

Copy `api.js` (provided below) into your React project:

**File location:** `src/api.js`

This file exports the `checkWashTradingRisk()` function that communicates with your Python backend.

## Step 4: Add the UI Component

Copy `App.js` (provided below) to your React project:

**File location:** `src/App.js`

This component provides a form to test the wash trading detector.

## Step 5: Run the React App

```bash
npm start
# or for Vite
npm run dev
```

Your app will open at `http://localhost:3000` (or `http://localhost:5173` for Vite).

## Step 6: Test It!

1. Enter a transaction price, time gap, and gas fee
2. Click "Check Risk Score"
3. The AI model will analyze and return:
   - `is_wash_trade`: boolean
   - `risk_score`: float (negative = more suspicious)
   - `status`: "HIGH RISK" or "SAFE"

## Troubleshooting

### CORS Error?
If you see CORS errors in the browser console, the backend is not allowing your frontend origin. The backend is already configured to allow:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

### Backend Not Responding?
Make sure:
1. The Python server is running on port 8000 (check `netstat -ano | findstr :8000`)
2. Both terminals are open (one for Python, one for React)
3. The backend prints "[OK] Brain Loaded Successfully" on startup

### Model Not Loaded Error?
If the `/analyze` endpoint returns `{"error": "Model not loaded"}`, restart the backend:
```powershell
# Kill old process and restart
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Start-Sleep 2
.venv\Scripts\python -m uvicorn ai_engine.api:app --port 8000
```

## Real Integration

To integrate into an existing NFT marketplace (like Sevantika):

1. **Copy `api.js`** to `src/services/api.js` in your project
2. **Import the function** in your NFT detail/buy component:
   ```javascript
   import { checkWashTradingRisk } from '../services/api';
   ```
3. **Call it before purchase:**
   ```javascript
   const handleBuyClick = async () => {
     try {
       const result = await checkWashTradingRisk({
         price_usd: nft.price,
         time_since_last_trade: getTimeSinceLastTrade(nft),
         sellerFee_amount: nft.fee
       });
       
       if (result.is_wash_trade) {
         alert(`⚠️ WARNING: HIGH WASH TRADING RISK! Score: ${result.risk_score}`);
         return; // Block the purchase
       }
       
       // Proceed with purchase
       proceedWithPayment();
     } catch (err) {
       console.error('Risk check failed:', err);
     }
   };
   ```
