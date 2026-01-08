import os
import google.genai as genai
from dotenv import load_dotenv
import json
import re

class SentimentAnalyzer:
    """
    Analyzes the sentiment of financial news using the Google Gemini API.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("GOOGLE_API_KEY not found or not set in .env file.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash', api_key=api_key)
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        return """
        Analyze the sentiment of the following financial news article from an investor's perspective for the company {ticker}. 
        Your response MUST be a single, valid JSON object with two keys:
        1. "score": A float from -1.0 (very negative) to 1.0 (very positive).
        2. "justification": A brief, one-sentence explanation for the score.

        **Article Title:** "{title}"
        **Article Description:** "{description}"

        **JSON Output:**
        """

    def analyze_sentiment(self, ticker: str, title: str, description: str) -> dict:
        try:
            prompt = self.prompt_template.format(ticker=ticker, title=title, description=description)
            response = self.model.generate_content(prompt)
            
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                print(f"WARNING: No JSON object found in LLM response. Text: {response.text}")
                return {"score": 0.0, "justification": "Parsing error."}

        except Exception as e:
            print(f"ERROR: Sentiment analysis failed: {e}")
            return {"score": 0.0, "justification": "API or processing error."}

if __name__ == '__main__':
    try:
        analyzer = SentimentAnalyzer()
        print("--- Testing Sentiment Analyzer (Final) ---")
        
        test_case = {"ticker": "TSLA", "title": "Tesla Announces Record Deliveries", "description": "Tesla Inc. announced record-breaking vehicle deliveries for the fourth quarter, exceeding analyst expectations and causing a surge in its stock price."}
        result = analyzer.analyze_sentiment(test_case["ticker"], test_case["title"], test_case["description"])
        
        print(f"\n--- Analyzing: '{test_case['title']}' ---")
        print(json.dumps(result, indent=2))
        
        assert isinstance(result['score'], float)
        print("\nTest passed.")

    except ValueError as e:
        print(f"CRITICAL: {e}. Please ensure your .env file is correct.")
