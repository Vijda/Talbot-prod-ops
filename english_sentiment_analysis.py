import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnglishSentimentAnalyzer:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the English sentiment analyzer with Ollama.
        
        Args:
            model_name: The Ollama model to use (default: llama3.2)
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def classify_sentiment_english(self, text: str) -> Dict[str, str]:
        """
        Classify the sentiment of a given text using Ollama and return English explanation.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment classification and English explanation
        """
        prompt = f"""
        Analyze the sentiment of the following review and provide a detailed English explanation.
        
        Review: "{text}"
        
        Please respond with:
        CLASSIFICATION: [POSITIVE/NEGATIVE/NEUTRAL]
        REASON: [Detailed explanation in English about why this sentiment was chosen]
        KEY_POINTS: [List 2-3 key points from the review in English]
        """
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Parse the response
                if 'POSITIVE' in response_text.upper():
                    sentiment = 'POSITIVE'
                elif 'NEGATIVE' in response_text.upper():
                    sentiment = 'NEGATIVE'
                else:
                    sentiment = 'NEUTRAL'
                
                return {
                    'sentiment': sentiment,
                    'confidence': 'high',
                    'english_reason': response_text,
                    'raw_response': response_text
                }
            else:
                return {
                    'sentiment': 'NEUTRAL',
                    'confidence': 'low',
                    'english_reason': 'API call failed',
                    'raw_response': 'Error: API call failed'
                }
                
        except Exception as e:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 'low',
                'english_reason': f'Error: {str(e)}',
                'raw_response': f'Error: {str(e)}'
            }
    
    def extract_issues_english(self, text: str) -> List[str]:
        """
        Extract potential issues from the review text in English.
        
        Args:
            text: The review text
            
        Returns:
            List of extracted issues in English
        """
        prompt = f"""
        Extract specific issues or problems mentioned in this review and translate them to English.
        If no issues are mentioned, respond with "No issues found".
        
        Review: "{text}"
        
        Please list the issues in English, separated by commas. If no issues, just say "No issues found".
        """
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                if 'no issues found' in response_text.lower():
                    return []
                else:
                    # Extract issues from response
                    issues = [issue.strip() for issue in response_text.split(',') if issue.strip()]
                    return issues
            else:
                return []
                
        except Exception as e:
            return []

class EnglishReviewAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the English review analyzer.
        
        Args:
            file_path: Path to the Excel file containing reviews
        """
        self.file_path = file_path
        self.df = pd.DataFrame()
        self.sentiment_analyzer = EnglishSentimentAnalyzer()
        
    def load_data(self) -> pd.DataFrame:
        """Load the review data from Excel file."""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded {len(self.df)} reviews from {self.file_path}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()
            return self.df
    
    def analyze_sentiments_english(self, batch_size: int = 10) -> pd.DataFrame:
        """
        Analyze sentiments for all reviews with English explanations.
        
        Args:
            batch_size: Number of reviews to process in each batch
            
        Returns:
            DataFrame with sentiment analysis results in English
        """
        if self.df.empty:
            return pd.DataFrame()
        
        # Check Ollama connection
        if not self.sentiment_analyzer.check_ollama_connection():
            print("Warning: Ollama is not running. Please start Ollama with llama3.2 model.")
            return self.df
        
        print("Starting English sentiment analysis...")
        
        sentiments = []
        confidences = []
        english_reasons = []
        raw_responses = []
        issues = []
        
        total_reviews = len(self.df)
        
        for i, (idx, row) in enumerate(self.df.iterrows()):
            # Use translated review if available, otherwise original
            review_text = row.get('Translated Review', row.get('Review', ''))
            
            review_text_str = str(review_text) if review_text is not None else ""
            if pd.isna(review_text) or review_text_str.strip() == '':
                sentiments.append('NEUTRAL')
                confidences.append('low')
                english_reasons.append('Empty review')
                raw_responses.append('Empty review')
                issues.append([])
                continue
            
            print(f"Processing review {i+1}/{total_reviews}: {review_text_str[:50]}...")
            
            # Analyze sentiment with English explanation
            sentiment_result = self.sentiment_analyzer.classify_sentiment_english(review_text_str)
            sentiments.append(sentiment_result['sentiment'])
            confidences.append(sentiment_result['confidence'])
            english_reasons.append(sentiment_result['english_reason'])
            raw_responses.append(sentiment_result['raw_response'])
            
            # Extract issues in English
            extracted_issues = self.sentiment_analyzer.extract_issues_english(review_text_str)
            issues.append(extracted_issues)
            
            # Progress update
            if (i + 1) % batch_size == 0:
                print(f"Processed {i+1}/{total_reviews} reviews...")
        
        # Add results to dataframe
        self.df['Sentiment'] = sentiments
        self.df['Confidence'] = confidences
        self.df['English_Reason'] = english_reasons
        self.df['Raw_Response'] = raw_responses
        self.df['English_Issues'] = issues
        
        return self.df
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the sentiment analysis."""
        if self.df.empty:
            return {}
        
        # Sentiment distribution
        sentiment_counts = self.df['Sentiment'].value_counts()
        
        # Confidence distribution
        confidence_counts = self.df['Confidence'].value_counts()
        
        # Calculate percentages
        total_reviews = len(self.df)
        sentiment_percentages = (sentiment_counts / total_reviews * 100).round(2)
        
        # Extract all issues
        all_issues = []
        for issues_list in self.df['English_Issues']:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        # Count issue frequency
        issue_counts = Counter(all_issues)
        top_issues = issue_counts.most_common(10)
        
        return {
            'total_reviews': total_reviews,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': sentiment_percentages.to_dict(),
            'confidence_distribution': confidence_counts.to_dict(),
            'top_issues': top_issues
        }
    
    def detect_trending_issues(self, min_frequency: int = 2) -> List[Tuple[str, int]]:
        """
        Detect trending issues based on frequency.
        
        Args:
            min_frequency: Minimum frequency to consider an issue as trending
            
        Returns:
            List of trending issues with their frequencies
        """
        if self.df.empty:
            return []
        
        # Extract all issues
        all_issues = []
        for issues_list in self.df['English_Issues']:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        # Count issue frequency
        issue_counts = Counter(all_issues)
        
        # Filter by minimum frequency
        trending_issues = [(issue, count) for issue, count in issue_counts.items() 
                          if count >= min_frequency]
        
        # Sort by frequency (descending)
        trending_issues.sort(key=lambda x: x[1], reverse=True)
        
        return trending_issues
    
    def save_english_results(self, output_path: str = "english_sentiment_analysis_results.xlsx"):
        """Save the analysis results to Excel file with English explanations."""
        if self.df.empty:
            print("No data to save.")
            return
        
        # Create a summary sheet
        summary_data = []
        stats = self.generate_summary_statistics()
        
        summary_data.append(['Total Reviews Analyzed', stats['total_reviews']])
        summary_data.append([''])
        
        summary_data.append(['Sentiment Distribution'])
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = stats['sentiment_percentages'].get(sentiment, 0)
            summary_data.append([sentiment, f"{count} reviews ({percentage}%)"])
        
        summary_data.append([''])
        summary_data.append(['Top Trending Issues (English)'])
        trending_issues = self.detect_trending_issues()
        for issue, count in trending_issues[:10]:
            summary_data.append([issue, f"{count} mentions"])
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data with English explanations
            self.df.to_excel(writer, sheet_name='Reviews with English Sentiment', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='English Summary', index=False)
        
        print(f"English results saved to {output_path}")

def main():
    """Main function to run the English sentiment analysis."""
    print("=== Talabat Reviews English Sentiment Analysis ===")
    print("Using Ollama with Llama3.2 for English sentiment classification")
    print()
    
    # Initialize analyzer
    analyzer = EnglishReviewAnalyzer("Talabat_Reviews.xlsx")
    
    # Load data
    df = analyzer.load_data()
    if df.empty:
        print("No data loaded. Please check the file path.")
        return
    
    # Check Ollama connection
    if not analyzer.sentiment_analyzer.check_ollama_connection():
        print("Error: Ollama is not running or not accessible.")
        print("Please start Ollama with the llama3.2 model:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull the model: ollama pull llama3.2")
        print("3. Start Ollama service")
        return
    
    print("Ollama connection successful!")
    print()
    
    # Analyze sentiments with English explanations
    print("Starting English sentiment analysis...")
    results_df = analyzer.analyze_sentiments_english(batch_size=5)
    
    if results_df.empty:
        print("No results to analyze.")
        return
    
    # Generate summary
    print("\n=== English Analysis Summary ===")
    stats = analyzer.generate_summary_statistics()
    
    print(f"Total Reviews Analyzed: {stats['total_reviews']}")
    print("\nSentiment Distribution:")
    for sentiment, percentage in stats['sentiment_percentages'].items():
        count = stats['sentiment_distribution'][sentiment]
        print(f"  {sentiment}: {count} reviews ({percentage}%)")
    
    # Detect trending issues
    print("\n=== Trending Issues (English) ===")
    trending_issues = analyzer.detect_trending_issues(min_frequency=2)
    
    if trending_issues:
        print("Top trending issues (mentioned 2+ times):")
        for issue, count in trending_issues[:10]:
            print(f"  • {issue}: {count} mentions")
    else:
        print("No trending issues detected (minimum frequency: 2)")
    
    # Save results
    print("\nSaving English results...")
    analyzer.save_english_results()
    
    print("\n=== Analysis Complete ===")
    print("English results saved to:")
    print("  - english_sentiment_analysis_results.xlsx (detailed results with English explanations)")

if __name__ == "__main__":
    main() 