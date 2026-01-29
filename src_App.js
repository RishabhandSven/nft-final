// src/App.js
// Example React component for testing the wash trading detector

import React, { useState, useEffect } from 'react';
import { checkWashTradingRisk, getAPIHealth } from './services/api';
import './App.css';

function App() {
  // Form state
  const [formData, setFormData] = useState({
    price_usd: 5000,
    time_since_last_trade: 60,
    sellerFee_amount: 5
  });

  // UI state
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on component mount
  useEffect(() => {
    checkAPIHealth();
    const interval = setInterval(checkAPIHealth, 5000); // Check every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const checkAPIHealth = async () => {
    const health = await getAPIHealth();
    if (health.status === 'ok' && health.model_loaded) {
      setApiStatus('ready');
    } else {
      setApiStatus('offline');
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (apiStatus !== 'ready') {
      setError('⚠️ Backend is not ready. Make sure Python server is running on port 8000');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await checkWashTradingRisk(formData);
      
      if (response.error) {
        setError(`API Error: ${response.error}`);
      } else {
        setResult(response);
      }
    } catch (err) {
      setError('Failed to analyze. Is the backend running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickTest = (scenario) => {
    switch (scenario) {
      case 'normal':
        setFormData({ price_usd: 500, time_since_last_trade: 86400, sellerFee_amount: 10 });
        break;
      case 'suspicious':
        setFormData({ price_usd: 5000, time_since_last_trade: 30, sellerFee_amount: 0 });
        break;
      case 'highvolume':
        setFormData({ price_usd: 50000, time_since_last_trade: 5, sellerFee_amount: 0.1 });
        break;
      default:
        break;
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🕵️ NFT Wash Trading Detector</h1>
        <p className="subtitle">AI-powered risk analysis powered by Isolation Forest</p>
        <div className={`api-status ${apiStatus}`}>
          {apiStatus === 'ready' && '🟢 Backend Ready'}
          {apiStatus === 'checking' && '🟡 Checking...'}
          {apiStatus === 'offline' && '🔴 Backend Offline'}
        </div>
      </header>

      <main className="app-main">
        {/* Form Section */}
        <section className="form-section">
          <h2>Analyze Transaction</h2>
          
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="price_usd">
                <strong>NFT Price (USD)</strong>
              </label>
              <input
                id="price_usd"
                type="number"
                name="price_usd"
                value={formData.price_usd}
                onChange={handleChange}
                step="0.01"
                min="0"
                placeholder="Enter price in USD"
              />
            </div>

            <div className="form-group">
              <label htmlFor="time_since_last_trade">
                <strong>Time Since Last Trade (seconds)</strong>
              </label>
              <small className="help-text">
                How long ago was this asset last traded? (30s = suspicious, 1 day = 86400s)
              </small>
              <input
                id="time_since_last_trade"
                type="number"
                name="time_since_last_trade"
                value={formData.time_since_last_trade}
                onChange={handleChange}
                min="0"
                placeholder="Seconds since last trade"
              />
            </div>

            <div className="form-group">
              <label htmlFor="sellerFee_amount">
                <strong>Seller Fee (USD)</strong>
              </label>
              <input
                id="sellerFee_amount"
                type="number"
                name="sellerFee_amount"
                value={formData.sellerFee_amount}
                onChange={handleChange}
                step="0.01"
                min="0"
                placeholder="Fee amount"
              />
            </div>

            <button 
              type="submit" 
              disabled={loading || apiStatus !== 'ready'}
              className="submit-btn"
            >
              {loading ? '🔄 Analyzing...' : '🔍 Check Risk Score'}
            </button>
          </form>

          {/* Quick Test Buttons */}
          <div className="quick-tests">
            <h3>Quick Tests:</h3>
            <button 
              onClick={() => handleQuickTest('normal')} 
              className="test-btn test-normal"
              disabled={loading}
            >
              Normal Trade ($500, 1 day gap)
            </button>
            <button 
              onClick={() => handleQuickTest('suspicious')} 
              className="test-btn test-suspicious"
              disabled={loading}
            >
              Suspicious ($5000, 30s gap)
            </button>
            <button 
              onClick={() => handleQuickTest('highvolume')} 
              className="test-btn test-high"
              disabled={loading}
            >
              High Volume ($50k, 5s gap)
            </button>
          </div>
        </section>

        {/* Error Display */}
        {error && (
          <section className="error-section">
            <p className="error-text">❌ {error}</p>
          </section>
        )}

        {/* Results Display */}
        {result && !result.error && (
          <section className={`result-section ${result.is_wash_trade ? 'risk-high' : 'risk-low'}`}>
            <h2>Analysis Result</h2>
            
            <div className="result-grid">
              <div className="result-item">
                <span className="label">Verdict:</span>
                <span className={`value verdict ${result.is_wash_trade ? 'dangerous' : 'safe'}`}>
                  {result.is_wash_trade ? '🚨 WASH TRADE DETECTED' : '✅ SAFE TRANSACTION'}
                </span>
              </div>
              
              <div className="result-item">
                <span className="label">Status:</span>
                <span className="value">{result.status}</span>
              </div>
              
              <div className="result-item">
                <span className="label">Risk Score:</span>
                <span className="value score">{result.risk_score.toFixed(6)}</span>
              </div>
            </div>

            <div className="risk-explanation">
              <p className="explanation-label">📊 How to read the score:</p>
              <ul>
                <li><strong>Negative scores</strong> = More likely to be anomalous (wash trade)</li>
                <li><strong>Positive scores</strong> = More likely normal trading pattern</li>
                <li><strong>Closer to 0</strong> = Borderline behavior</li>
              </ul>
            </div>

            <div className="transaction-summary">
              <p><strong>Input Transaction:</strong></p>
              <pre>
{JSON.stringify({
  price_usd: formData.price_usd,
  time_since_last_trade: formData.time_since_last_trade,
  sellerFee_amount: formData.sellerFee_amount
}, null, 2)}
              </pre>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>
          💡 Tip: Keep the Python backend running in a separate terminal:
          <br />
          <code>.venv\Scripts\python -m uvicorn ai_engine.api:app --port 8000</code>
        </p>
      </footer>
    </div>
  );
}

export default App;
