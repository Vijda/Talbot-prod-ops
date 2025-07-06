import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import requests
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class OllamaSentimentAnalyzer:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the sentiment analyzer with Ollama.
        
        Args:
            model_name: The Ollama model to use (default: llama3)
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
    
    def classify_sentiment(self, text: str) -> Dict[str, str]:
        """
        Classify the sentiment of a given text using Ollama.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment classification and confidence
        """
        prompt = f"""
        Analyze the sentiment of the following review and classify it as POSITIVE, NEGATIVE, or NEUTRAL.
        
        Review: "{text}"
        
        Please respond with only the classification (POSITIVE/NEGATIVE/NEUTRAL) and a brief reason in English.
        Format: CLASSIFICATION: [POSITIVE/NEGATIVE/NEUTRAL] | REASON: [brief explanation in English]
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
                    'reason': response_text
                }
            else:
                return {
                    'sentiment': 'NEUTRAL',
                    'confidence': 'low',
                    'reason': 'API call failed'
                }
                
        except Exception as e:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 'low',
                'reason': f'Error: {str(e)}'
            }
    
    def extract_issues(self, text: str) -> List[str]:
        """
        Extract potential issues from the review text.
        
        Args:
            text: The review text
            
        Returns:
            List of extracted issues
        """
        prompt = f"""
        Extract specific issues or problems mentioned in this review. 
        If no issues are mentioned, respond with "No issues found". Also, consider yourself as an arabic expert, who understand the review.
        
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

class ReviewAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the review analyzer.
        
        Args:
            file_path: Path to the Excel file containing reviews
        """
        self.file_path = file_path
        self.df = pd.DataFrame()
        self.sentiment_analyzer = OllamaSentimentAnalyzer()
        
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
    
    def analyze_sentiments(self, batch_size: int = 10) -> pd.DataFrame:
        """
        Analyze sentiments for all reviews.
        
        Args:
            batch_size: Number of reviews to process in each batch
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if self.df is None:
            self.load_data()
        
        if self.df.empty:
            return pd.DataFrame()
        
        # Check Ollama connection
        if not self.sentiment_analyzer.check_ollama_connection():
            print("Warning: Ollama is not running. Please start Ollama with llama3 model.")
            return self.df
        
        print("Starting sentiment analysis...")
        
        sentiments = []
        confidences = []
        reasons = []
        issues = []
        
        total_reviews = len(self.df)
        
        for i, (idx, row) in enumerate(self.df.iterrows()):
            # Use translated review if available, otherwise original
            review_text = row.get('Translated Review', row.get('Review', ''))
            
            review_text_str = str(review_text) if review_text is not None else ""
            if pd.isna(review_text) or review_text_str.strip() == '':
                sentiments.append('NEUTRAL')
                confidences.append('low')
                reasons.append('Empty review')
                issues.append([])
                continue
            
            print(f"Processing review {i+1}/{total_reviews}: {review_text_str[:50]}...")
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.classify_sentiment(review_text_str)
            sentiments.append(sentiment_result['sentiment'])
            confidences.append(sentiment_result['confidence'])
            reasons.append(sentiment_result['reason'])
            
            # Extract issues
            extracted_issues = self.sentiment_analyzer.extract_issues(review_text_str)
            issues.append(extracted_issues)
            
            # Progress update
            if (i + 1) % batch_size == 0:
                print(f"Processed {i+1}/{total_reviews} reviews...")
        
        # Add results to dataframe
        self.df['Sentiment'] = sentiments
        self.df['Confidence'] = confidences
        self.df['Reason'] = reasons
        self.df['Issues'] = issues
        
        return self.df
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the sentiment analysis."""
        if self.df is None or self.df.empty:
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
        for issues_list in self.df['Issues']:
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
        if self.df is None or self.df.empty:
            return []
        
        # Extract all issues
        all_issues = []
        for issues_list in self.df['Issues']:
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
    
    def create_visualizations(self, save_path: str = "sentiment_analysis_results.png"):
        """Create and save visualizations for the analysis."""
        if self.df is None or self.df.empty:
            print("No data to visualize.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = self.df['Sentiment'].value_counts()
        colors = ['#2E8B57', '#DC143C', '#FFD700']  # Green, Red, Gold
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Sentiment Distribution')
        
        # 2. Sentiment Bar Chart
        axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values, 
                       color=colors[:len(sentiment_counts)])
        axes[0, 1].set_title('Sentiment Counts')
        axes[0, 1].set_ylabel('Number of Reviews')
        
        # 3. Confidence Distribution
        confidence_counts = self.df['Confidence'].value_counts()
        axes[1, 0].bar(confidence_counts.index, confidence_counts.values, 
                       color=['#4CAF50', '#FF9800', '#F44336'])
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_ylabel('Number of Reviews')
        
        # 4. Top Issues (if any)
        all_issues = []
        for issues_list in self.df['Issues']:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)
        
        if all_issues:
            issue_counts = Counter(all_issues)
            top_issues = issue_counts.most_common(8)
            
            if top_issues:
                issues, counts = zip(*top_issues)
                axes[1, 1].barh(range(len(issues)), counts, color='#2196F3')
                axes[1, 1].set_yticks(range(len(issues)))
                axes[1, 1].set_yticklabels(issues)
                axes[1, 1].set_title('Top Issues')
                axes[1, 1].set_xlabel('Frequency')
        else:
            axes[1, 1].text(0.5, 0.5, 'No issues detected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top Issues')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {save_path}")
    
    def save_results(self, output_path: str = "sentiment_analysis_results.xlsx"):
        """Save the analysis results to Excel file."""
        if self.df is None or self.df.empty:
            print("No data to save.")
            return
        
        # Create a summary sheet
        summary_data = []
        stats = self.generate_summary_statistics()
        
        summary_data.append(['Total Reviews', stats['total_reviews']])
        summary_data.append([''])
        
        summary_data.append(['Sentiment Distribution'])
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = stats['sentiment_percentages'].get(sentiment, 0)
            summary_data.append([sentiment, f"{count} ({percentage}%)"])
        
        summary_data.append([''])
        summary_data.append(['Top Trending Issues'])
        trending_issues = self.detect_trending_issues()
        for issue, count in trending_issues[:10]:
            summary_data.append([issue, count])
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data
            self.df.to_excel(writer, sheet_name='Reviews with Sentiment', index=False)
            
            # Summary sheet
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Results saved to {output_path}")

def main():
    """Main function to run the sentiment analysis."""
    print("=== Talabat Reviews Sentiment Analysis ===")
    print("Using Ollama with Llama3 for sentiment classification")
    print()
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer("Talabat Reviews.xlsx")
    
    # Load data
    df = analyzer.load_data()
    if df.empty:
        print("No data loaded. Please check the file path.")
        return
    
    # Check Ollama connection
    if not analyzer.sentiment_analyzer.check_ollama_connection():
        print("Error: Ollama is not running or not accessible.")
        print("Please start Ollama with the llama3 model:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull the model: ollama pull llama3")
        print("3. Start Ollama service")
        return
    
    print("Ollama connection successful!")
    print()
    
    # Analyze sentiments
    print("Starting sentiment analysis...")
    results_df = analyzer.analyze_sentiments(batch_size=5)
    
    if results_df.empty:
        print("No results to analyze.")
        return
    
    # Generate summary
    print("\n=== Analysis Summary ===")
    stats = analyzer.generate_summary_statistics()
    
    print(f"Total Reviews Analyzed: {stats['total_reviews']}")
    print("\nSentiment Distribution:")
    for sentiment, percentage in stats['sentiment_percentages'].items():
        count = stats['sentiment_distribution'][sentiment]
        print(f"  {sentiment}: {count} reviews ({percentage}%)")
    
    # Detect trending issues
    print("\n=== Trending Issues ===")
    trending_issues = analyzer.detect_trending_issues(min_frequency=2)
    
    if trending_issues:
        print("Top trending issues (mentioned 2+ times):")
        for issue, count in trending_issues[:10]:
            print(f"  • {issue}: {count} mentions")
    else:
        print("No trending issues detected (minimum frequency: 2)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations()
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results()
    
    print("\n=== Analysis Complete ===")
    print("Results saved to:")
    print("  - sentiment_analysis_results.xlsx (detailed results)")
    print("  - sentiment_analysis_results.png (visualizations)")

if __name__ == "__main__":
    main() 