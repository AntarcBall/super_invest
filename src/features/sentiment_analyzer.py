import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

class SentimentAnalyzer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("GOOGLE_API_KEY not found or not set in .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        return """
        As a Financial Analyst, analyze the sentiment of the following news article 
        from an investor's perspective. Provide your output in a clean JSON format with two keys: "score" 
        (a float from -1.0 to 1.0) and "justification" (a brief explanation).

        ---
        **Example 1:**
        **Article:** "Tesla's Q4 Earnings Exceed Expectations, Shares Surge 10%"
        **Output:**
        ```json
        {
          "score": 0.8,
          "justification": "Reports stronger-than-expected earnings and positive market reaction."
        }
        ```

        **Example 2:**
        **Article:** "Investigation launched into Meta's data privacy practices."
        **Output:**
        ```json
        {
          "score": -0.7,
          "justification": "Highlights a significant regulatory risk and potential legal challenges."
        }
        ```
        ---

        **Analyze:**
        **Title:** "{title}"
        **Description:** "{description}"
        
        **Output:**
        """

    def analyze_sentiment(self, title: str, description: str) -> dict:
        try:
            prompt = self.prompt_template.format(title=title, description=description)
            response = self.model.generate_content(prompt)
            
            # Use regex to find the JSON block, which is more robust
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in the response.")

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"ERROR: Could not parse LLM response: {e}")
            return {"score": 0.0, "justification": "Error during analysis."}
        except Exception as e:
            print(f"ERROR: Sentiment analysis failed: {e}")
            return {"score": 0.0, "justification": "Error during analysis."}

if __name__ == '__main__':
    try:
        analyzer = SentimentAnalyzer()
        
        print("--- Testing Sentiment Analyzer ---")
        test_cases = [
            {"title": "Apple sales boom", "description": "Record-breaking iPhone sales reported, shares jump 5%."},
            {"title": "Ford recalls cars", "description": "Major recall of 1.5 million vehicles due to brake fluid leaks, stock drops."}
        ]

        for case in test_cases:
            result = analyzer.analyze_sentiment(case["title"], case["description"])
            print(f"\n--- Analyzing: '{case['title']}' ---")
            print(json.dumps(result, indent=2))

    except ValueError as e:
        print(f"CRITICAL: {e}. Please ensure your .env file is correct and has a valid GOOGLE_API_KEY.")
