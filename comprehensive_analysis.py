import pandas as pd
import numpy as np
import requests
import json
from collections import Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the comprehensive analyzer with Ollama.
        
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
        Consider yourself as an Arabic expert who understands the review context.
        
        Review: "{text}"
        
        Please respond with:
        CLASSIFICATION: [POSITIVE/NEGATIVE/NEUTRAL]
        REASON: [Detailed explanation in English about why this sentiment was chosen]
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
        Consider yourself as an Arabic expert who understands the review context.
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
    
    def extract_positive_aspects(self, text: str) -> List[str]:
        """
        Extract positive aspects from the review text in English.
        
        Args:
            text: The review text
            
        Returns:
            List of positive aspects in English
        """
        prompt = f"""
        Extract positive aspects or good things mentioned in this review and translate them to English.
        Consider yourself as an Arabic expert who understands the review context.
        If no positive aspects are mentioned, respond with "No positive aspects found".
        
        Review: "{text}"
        
        Please list the positive aspects in English, separated by commas. If no positive aspects, just say "No positive aspects found".
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
                
                if 'no positive aspects found' in response_text.lower():
                    return []
                else:
                    # Extract positive aspects from response
                    aspects = [aspect.strip() for aspect in response_text.split(',') if aspect.strip()]
                    return aspects
            else:
                return []
                
        except Exception as e:
            return []
    
    def extract_suggestions(self, text: str) -> List[str]:
        """
        Extract customer suggestions from the review text in English.
        
        Args:
            text: The review text
            
        Returns:
            List of suggestions in English
        """
        prompt = f"""
        Extract suggestions, recommendations, or improvements mentioned by the customer in this review and translate them to English.
        Consider yourself as an Arabic expert who understands the review context.
        If no suggestions are mentioned, respond with "No suggestions found".
        
        Review: "{text}"
        
        Please list the suggestions in English, separated by commas. If no suggestions, just say "No suggestions found".
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
                
                if 'no suggestions found' in response_text.lower():
                    return []
                else:
                    # Extract suggestions from response
                    suggestions = [suggestion.strip() for suggestion in response_text.split(',') if suggestion.strip()]
                    return suggestions
            else:
                return []
                
        except Exception as e:
            return []

class ReviewComprehensiveAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the comprehensive review analyzer.
        
        Args:
            file_path: Path to the Excel file containing reviews
        """
        self.file_path = file_path
        self.df = pd.DataFrame()
        self.analyzer = ComprehensiveAnalyzer()
        
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
    
    def analyze_comprehensive(self, batch_size: int = 5) -> pd.DataFrame:
        """
        Perform comprehensive analysis of all reviews.
        
        Args:
            batch_size: Number of reviews to process in each batch
            
        Returns:
            DataFrame with comprehensive analysis results
        """
        if self.df.empty:
            return pd.DataFrame()
        
        # Check Ollama connection
        if not self.analyzer.check_ollama_connection():
            print("Warning: Ollama is not running. Please start Ollama with llama3.2 model.")
            return self.df
        
        print("Starting comprehensive analysis...")
        
        sentiments = []
        confidences = []
        english_reasons = []
        raw_responses = []
        issues = []
        positive_aspects = []
        suggestions = []
        
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
                positive_aspects.append([])
                suggestions.append([])
                continue
            
            print(f"Processing review {i+1}/{total_reviews}: {review_text_str[:50]}...")
            
            # Analyze sentiment with English explanation
            sentiment_result = self.analyzer.classify_sentiment_english(review_text_str)
            sentiments.append(sentiment_result['sentiment'])
            confidences.append(sentiment_result['confidence'])
            english_reasons.append(sentiment_result['english_reason'])
            raw_responses.append(sentiment_result['raw_response'])
            
            # Extract issues in English
            extracted_issues = self.analyzer.extract_issues_english(review_text_str)
            issues.append(extracted_issues)
            
            # Extract positive aspects
            extracted_positive = self.analyzer.extract_positive_aspects(review_text_str)
            positive_aspects.append(extracted_positive)
            
            # Extract suggestions
            extracted_suggestions = self.analyzer.extract_suggestions(review_text_str)
            suggestions.append(extracted_suggestions)
            
            # Progress update
            if (i + 1) % batch_size == 0:
                print(f"Processed {i+1}/{total_reviews} reviews...")
        
        # Add results to dataframe
        self.df['Sentiment'] = sentiments
        self.df['Confidence'] = confidences
        self.df['English_Reason'] = english_reasons
        self.df['Raw_Response'] = raw_responses
        self.df['English_Issues'] = issues
        self.df['Positive_Aspects'] = positive_aspects
        self.df['Customer_Suggestions'] = suggestions
        
        return self.df
    
    def get_top_issues(self, top_n: int = 4) -> List[Tuple[str, int]]:
        """Get top N issues mentioned across all reviews."""
        all_issues = []
        for issues_list in self.df['English_Issues']:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        issue_counts = Counter(all_issues)
        return issue_counts.most_common(top_n)
    
    def get_top_positive_aspects(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get top N positive aspects mentioned across all reviews."""
        all_positive = []
        for aspects_list in self.df['Positive_Aspects']:
            if isinstance(aspects_list, list):
                all_positive.extend(aspects_list)
        
        positive_counts = Counter(all_positive)
        return positive_counts.most_common(top_n)
    
    def get_top_suggestions(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get top N suggestions mentioned across all reviews."""
        all_suggestions = []
        for suggestions_list in self.df['Customer_Suggestions']:
            if isinstance(suggestions_list, list):
                all_suggestions.extend(suggestions_list)
        
        suggestion_counts = Counter(all_suggestions)
        return suggestion_counts.most_common(top_n)
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the analysis."""
        if self.df.empty:
            return {}
        
        # Sentiment distribution
        sentiment_counts = self.df['Sentiment'].value_counts()
        
        # Calculate percentages
        total_reviews = len(self.df)
        sentiment_percentages = (sentiment_counts / total_reviews * 100).round(2)
        
        return {
            'total_reviews': total_reviews,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': sentiment_percentages.to_dict(),
            'top_issues': self.get_top_issues(4),
            'top_positive_aspects': self.get_top_positive_aspects(3),
            'top_suggestions': self.get_top_suggestions(3)
        }
    
    def save_comprehensive_results(self, output_path: str = "comprehensive_analysis_results.xlsx"):
        """Save the comprehensive analysis results to Excel file."""
        if self.df.empty:
            print("No data to save.")
            return
        
        # Create summary data
        summary_data = []
        stats = self.generate_summary_statistics()
        
        summary_data.append(['Total Reviews Analyzed', stats['total_reviews']])
        summary_data.append([''])
        
        summary_data.append(['Sentiment Distribution'])
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = stats['sentiment_percentages'].get(sentiment, 0)
            summary_data.append([sentiment, f"{count} reviews ({percentage}%)"])
        
        summary_data.append([''])
        summary_data.append(['Top 4 Issues (Most Repeated)'])
        for issue, count in stats['top_issues']:
            summary_data.append([issue, f"{count} mentions"])
        
        summary_data.append([''])
        summary_data.append(['Top 3 Positive Aspects'])
        for aspect, count in stats['top_positive_aspects']:
            summary_data.append([aspect, f"{count} mentions"])
        
        summary_data.append([''])
        summary_data.append(['Top 3 Customer Suggestions'])
        for suggestion, count in stats['top_suggestions']:
            summary_data.append([suggestion, f"{count} mentions"])
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data with comprehensive analysis
            self.df.to_excel(writer, sheet_name='Reviews with Analysis', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Comprehensive results saved to {output_path}")

def main():
    """Main function to run the comprehensive analysis."""
    print("=== Talabat Reviews Comprehensive Analysis ===")
    print("Classifying reviews as Positive, Negative, and Neutral in English")
    print("Extracting top issues, positive aspects, and customer suggestions")
    print()
    
    # Initialize analyzer
    analyzer = ReviewComprehensiveAnalyzer("Talabat Reviews.xlsx")
    
    # Load data
    df = analyzer.load_data()
    if df.empty:
        print("No data loaded. Please check the file path.")
        return
    
    # Check Ollama connection
    if not analyzer.analyzer.check_ollama_connection():
        print("Error: Ollama is not running or not accessible.")
        print("Please start Ollama with the llama3.2 model:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull the model: ollama pull llama3.2")
        print("3. Start Ollama service")
        return
    
    print("Ollama connection successful!")
    print()
    
    # Perform comprehensive analysis
    print("Starting comprehensive analysis...")
    results_df = analyzer.analyze_comprehensive(batch_size=5)
    
    if results_df.empty:
        print("No results to analyze.")
        return
    
    # Generate summary
    print("\n=== Comprehensive Analysis Summary ===")
    stats = analyzer.generate_summary_statistics()
    
    print(f"Total Reviews Analyzed: {stats['total_reviews']}")
    print("\nSentiment Distribution:")
    for sentiment, percentage in stats['sentiment_percentages'].items():
        count = stats['sentiment_distribution'][sentiment]
        print(f"  {sentiment}: {count} reviews ({percentage}%)")
    
    # Show top issues
    print("\n=== Top 4 Issues (Most Repeated) ===")
    for issue, count in stats['top_issues']:
        print(f"  • {issue}: {count} mentions")
    
    # Show top positive aspects
    print("\n=== Top 3 Positive Aspects ===")
    for aspect, count in stats['top_positive_aspects']:
        print(f"  • {aspect}: {count} mentions")
    
    # Show top suggestions
    print("\n=== Top 3 Customer Suggestions ===")
    for suggestion, count in stats['top_suggestions']:
        print(f"  • {suggestion}: {count} mentions")
    
    # Save results
    print("\nSaving comprehensive results...")
    analyzer.save_comprehensive_results()
    
    print("\n=== Analysis Complete ===")
    print("Comprehensive results saved to:")
    print("  - comprehensive_analysis_results.xlsx (detailed analysis with all insights)")

if __name__ == "__main__":
    main() 