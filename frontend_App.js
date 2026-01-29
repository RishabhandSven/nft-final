import React, { useState, useEffect } from 'react';
import { checkWashTradingRisk, checkHealth } from './services/api';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    price_usd: 5000,
    time_since_last_trade: 3600,
    sellerFee_amount: 0,
    buyer_address: '0xBuyer001',
    seller_address: '0xSeller001',
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      const health = await checkHealth();
      setBackendHealth(health);
    };
    checkBackend();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await checkWashTradingRisk(formData);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Failed to analyze transaction');
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (preset) => {
    const presets = {
      safe: {
        price_usd: 5000,
        time_since_last_trade: 86400, // 1 day
        sellerFee_amount: 250,
        buyer_address: '0xBuyer001',
        seller_address: '0xSeller001',
      },
      suspicious: {
        price_usd: 5000,
        time_since_last_trade: 300, // 5 minutes
        sellerFee_amount: 50,
        buyer_address: '0xBuyer002',
        seller_address: '0xSeller002',
      },
      highRisk: {
        price_usd: 50000,
        time_since_last_trade: 30, // 30 seconds
        sellerFee_amount: 0,
        buyer_address: '0xBuyer003',
        seller_address: '0xSeller003',
      },
    };
    setFormData(presets[preset]);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔍 NFT Wash Trading Detection</h1>
        <p>AI-powered anomaly detection for suspicious NFT transactions</p>
        {backendHealth && (
          <div className={`health-status ${backendHealth.status === 'ok' ? 'healthy' : 'unhealthy'}`}>
            Backend: {backendHealth.status === 'ok' ? '✓ Connected' : '✗ Offline'}
          </div>
        )}
      </header>

      <main className="container">
        <div className="form-section">
          <h2>Transaction Details</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Price (USD)</label>
              <input
                type="number"
                name="price_usd"
                value={formData.price_usd}
                onChange={handleChange}
                step="0.01"
                min="0"
              />
            </div>

            <div className="form-group">
              <label>Time Since Last Trade (seconds)</label>
              <input
                type="number"
                name="time_since_last_trade"
                value={formData.time_since_last_trade}
                onChange={handleChange}
                min="0"
              />
            </div>

            <div className="form-group">
              <label>Seller Fee (USD)</label>
              <input
                type="number"
                name="sellerFee_amount"
                value={formData.sellerFee_amount}
                onChange={handleChange}
                step="0.01"
                min="0"
              />
            </div>

            <div className="form-group">
              <label>Buyer Address</label>
              <input
                type="text"
                name="buyer_address"
                value={formData.buyer_address}
                onChange={handleChange}
                placeholder="0x..."
              />
            </div>

            <div className="form-group">
              <label>Seller Address</label>
              <input
                type="text"
                name="seller_address"
                value={formData.seller_address}
                onChange={handleChange}
                placeholder="0x..."
              />
            </div>

            <div className="button-group">
              <button type="submit" disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze Transaction'}
              </button>
              <div className="preset-buttons">
                <button type="button" onClick={() => loadPreset('safe')} className="preset-btn safe">
                  Safe Example
                </button>
                <button type="button" onClick={() => loadPreset('suspicious')} className="preset-btn suspicious">
                  Suspicious
                </button>
                <button type="button" onClick={() => loadPreset('highRisk')} className="preset-btn highrisk">
                  High Risk
                </button>
              </div>
            </div>
          </form>
        </div>

        <div className="result-section">
          {error && (
            <div className="result error">
              <h3>❌ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <div className={`result ${result.status.toLowerCase().replace(' ', '-')}`}>
              <h3>{result.is_wash_trade ? '⚠️ WASH TRADE DETECTED' : '✅ TRANSACTION SAFE'}</h3>
              <div className="result-details">
                <div className="result-item">
                  <span className="label">Risk Assessment:</span>
                  <span className="value">{result.status}</span>
                </div>
                <div className="result-item">
                  <span className="label">Risk Score:</span>
                  <span className="value">{(result.risk_score * 100).toFixed(2)}%</span>
                </div>
                <div className="risk-bar">
                  <div 
                    className="risk-fill"
                    style={{
                      width: `${Math.max(0, Math.min(100, (result.risk_score + 1) * 50))}%`,
                      backgroundColor: result.is_wash_trade ? '#ff4444' : '#44ff44'
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="result placeholder">
              <p>Fill in transaction details and click "Analyze Transaction" to get started</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
