/**
 * API Service for NFT Wash Trading Detection
 * Communicates with Python backend at http://localhost:8000
 */

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const checkWashTradingRisk = async (transactionData) => {
  try {
    const payload = {
      price_usd: parseFloat(transactionData.price_usd) || 0,
      time_since_last_trade: parseInt(transactionData.time_since_last_trade) || 3600,
      sellerFee_amount: parseFloat(transactionData.sellerFee_amount) || 0,
      buyer_address: transactionData.buyer_address || '0x0',
      seller_address: transactionData.seller_address || '0x0',
    };

    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error calling API:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error('Error checking health:', error);
    return { status: 'error', model_loaded: false, scaler_loaded: false };
  }
};
