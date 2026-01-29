// src/services/api.js
// Service for communicating with the wash trading detection backend

const API_URL = "http://127.0.0.1:8000";

/**
 * Check if a transaction is likely a wash trade
 * @param {Object} transactionData - Transaction details
 * @param {number} transactionData.price_usd - NFT price in USD
 * @param {number} transactionData.time_since_last_trade - Seconds since last trade
 * @param {number} transactionData.sellerFee_amount - Seller fee in USD
 * @param {string} transactionData.buyer_address - Buyer wallet address (optional)
 * @param {string} transactionData.seller_address - Seller wallet address (optional)
 * @returns {Promise<Object>} { is_wash_trade, risk_score, status }
 */
export const checkWashTradingRisk = async (transactionData) => {
  try {
    // Map common field names to API expected format
    const payload = {
      price_usd: transactionData.price_usd || transactionData.price || 0,
      time_since_last_trade: transactionData.time_since_last_trade || transactionData.time_gap_seconds || 3600,
      sellerFee_amount: transactionData.sellerFee_amount || transactionData.gas_fee_usd || 0,
      buyer_address: transactionData.buyer_address || "0x0",
      seller_address: transactionData.seller_address || "0x0",
    };

    console.log("[API] Sending wash trade check:", payload);

    const response = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    console.log("[API] Response:", data);
    return data;
  } catch (error) {
    console.error("[API] Error:", error.message);
    // Return a safe default instead of throwing
    return {
      error: error.message,
      is_wash_trade: false,
      risk_score: 0,
      status: "ERROR"
    };
  }
};

/**
 * Get the health status of the API
 * @returns {Promise<Object>} { status, model_loaded, scaler_loaded }
 */
export const getAPIHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("[API] Health check failed:", error);
    return { status: "offline", model_loaded: false, scaler_loaded: false };
  }
};
