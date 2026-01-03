# 🏗️ Construction Company Evaluator

A smart tool that helps evaluate construction companies based on project requirements using AI.

## What Does It Do?

This project automatically:
1. **Reads** your construction project requirements (budget, timeline, scope)
2. **Searches** for information about a construction company (history, past projects, reputation)
3. **Analyzes** how well the company matches your project needs
4. **Gives** a confidence score on whether the company is a good fit

## How It Works

### `graph.py` - The Brain
- Uses **LangGraph** to organize the analysis process step by step
- Uses **Google's Gemini AI** to understand and extract project requirements
- Uses **Tavily Search** to find real information about companies online
- Structured workflow:
  - Extract project details from your requirements
  - Search for company information
  - Summarize what was found
  - Score how well the company fits

### `app.py` - The Interface
- Simple web interface built with **Streamlit**
- You enter:
  - Your project requirements (budget, timeline, scope)
  - Company name you want to evaluate
- The app shows:
  - Key company information
  - Overall confidence score
  - Why it gave that score
  - Links to sources

## How to Use

1. **Setup** - Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or 
   ```bash
   uv sync
   ```


2. **Run** - Start the web app:
   ```
   streamlit run app.py
   ```

3. **Enter** - Fill in your project details and company name

4. **Get Results** - See the evaluation instantly!

## Requirements

- Python 3.8+
- API Keys for:
  - Google Gemini
  - Tavily Search
  - LangSmith (for tracking)

## What You Get

✅ Automated company evaluation  
✅ AI-powered insights  
✅ Real web search results  
✅ Confidence scoring  
✅ Easy-to-use web interface
