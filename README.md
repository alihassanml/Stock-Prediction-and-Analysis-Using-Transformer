# **Stock Prediction and Analysis Using Transformer**

This project leverages **Transformers** for stock prediction and analysis, integrating **multi-agent systems** to provide real-time insights into stock data, financial news, and market trends.

## Features:

* Real-time stock price analysis
* Latest market news and trends
* Integrates financial data from **Yahoo Finance** (`YFinanceTools`)
* Uses  **Groq** 's **LLaMA 70B model** for enhanced AI predictions
* Multi-Agent architecture combining web search and finance tools

## Tech Stack:

* **FastAPI** : Web framework for building the API
* **Groq** : LLaMA 70B model for AI-powered predictions
* **YFinanceTools** : For stock data
* **DuckDuckGo** : For web-based search queries
* **Python-dotenv** : For environment variable management

## How to Run:

1. Clone the repository:

   ```bash
   git clone https://github.com/alihassanml/Stock-Prediction-and-Analysis-Using-Transformer.git
   cd Stock-Prediction-and-Analysis-Using-Transformer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Set up your **Groq API key** in a `.env` file:

   ```
   GROQ_API_KEY=your_groq_api_key
   ```
4. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```
5. Access the API at:

   `http://127.0.0.1:8000`
6. Query stock prices, financial news, and trends using the `/ask` endpoint:
   Example:

