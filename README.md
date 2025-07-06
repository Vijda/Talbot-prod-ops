# Talabat Reviews Sentiment Analysis

This project provides a comprehensive sentiment analysis system for Talabat reviews using Ollama with Llama3 for natural language processing.

## Features

- **Sentiment Classification**: Classify reviews as Positive, Negative, or Neutral
- **Issue Detection**: Extract and identify trending issues from reviews
- **Visualization**: Generate charts and graphs for analysis results
- **Excel Export**: Save detailed results to Excel files
- **Batch Processing**: Process reviews in batches for efficiency

## Prerequisites

1. **Python 3.8+**
2. **Ollama** with Llama3 model installed
3. **Required Python packages** (see requirements.txt)

## Setup Instructions

### 1. Install Ollama

Visit [https://ollama.ai/](https://ollama.ai/) and follow the installation instructions for your operating system.

### 2. Install Llama3 Model

```bash
ollama pull llama3
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Setup

Run the test script to verify everything is working:

```bash
python test_ollama_connection.py
```

You should see:
```
✅ Ollama is running and accessible!
✅ Llama3 model is available!
✅ Sentiment analysis test successful!
```

## Usage

### Basic Usage

1. **Prepare your data**: Ensure your Excel file (`Talabat Reviews.xlsx`) contains a column named `Review` with the review text.

2. **Run the analysis**:
   ```bash
   python sentiment_analysis.py
   ```

3. **View results**: The script will generate:
   - `sentiment_analysis_results.xlsx` - Detailed results with sentiment classifications
   - `sentiment_analysis_results.png` - Visualizations of the analysis

### Advanced Usage

You can also use the classes directly in your own scripts:

```python
from sentiment_analysis import ReviewAnalyzer

# Initialize analyzer
analyzer = ReviewAnalyzer("your_reviews.xlsx")

# Load and analyze data
df = analyzer.load_data()
results = analyzer.analyze_sentiments(batch_size=5)

# Generate summary statistics
stats = analyzer.generate_summary_statistics()
print(f"Total reviews: {stats['total_reviews']}")

# Detect trending issues
trending_issues = analyzer.detect_trending_issues(min_frequency=2)
for issue, count in trending_issues:
    print(f"{issue}: {count} mentions")

# Create visualizations
analyzer.create_visualizations("my_results.png")

# Save results
analyzer.save_results("my_results.xlsx")
```

## Output Files

### Excel Output (`sentiment_analysis_results.xlsx`)

Contains two sheets:

1. **Reviews with Sentiment**: Original data plus:
   - `Sentiment`: POSITIVE/NEGATIVE/NEUTRAL
   - `Confidence`: high/medium/low
   - `Reason`: Brief explanation of classification
   - `Issues`: List of extracted issues

2. **Summary**: Key statistics including:
   - Total reviews analyzed
   - Sentiment distribution percentages
   - Top trending issues

### Visualization Output (`sentiment_analysis_results.png`)

Four-panel visualization showing:
1. **Sentiment Distribution** (Pie Chart)
2. **Sentiment Counts** (Bar Chart)
3. **Confidence Distribution** (Bar Chart)
4. **Top Issues** (Horizontal Bar Chart)

## Configuration

### Batch Size

You can adjust the batch size for processing:

```python
# Process 10 reviews at a time (default)
results = analyzer.analyze_sentiments(batch_size=10)

# Process 5 reviews at a time (faster but more API calls)
results = analyzer.analyze_sentiments(batch_size=5)
```

### Trending Issues Threshold

Adjust the minimum frequency for trending issues:

```python
# Issues mentioned 2+ times (default)
trending_issues = analyzer.detect_trending_issues(min_frequency=2)

# Issues mentioned 5+ times
trending_issues = analyzer.detect_trending_issues(min_frequency=5)
```

## Troubleshooting

### Ollama Connection Issues

1. **Ollama not running**:
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Model not found**:
   ```bash
   # Pull the required model
   ollama pull llama3
   ```

3. **Connection refused**:
   - Check if Ollama is running on port 11434
   - Verify firewall settings
   - Try restarting Ollama service

### Performance Issues

1. **Slow processing**: Reduce batch size
2. **Memory issues**: Process smaller datasets
3. **API timeouts**: Increase timeout settings in the script

### Data Issues

1. **Empty reviews**: The script handles empty reviews automatically
2. **Encoding issues**: Ensure Excel file is saved with UTF-8 encoding
3. **Missing columns**: Verify your Excel file has a `Review` column

## API Reference

### OllamaSentimentAnalyzer

Main class for sentiment analysis using Ollama.

```python
analyzer = OllamaSentimentAnalyzer(model_name="llama3", base_url="http://localhost:11434")

# Check connection
if analyzer.check_ollama_connection():
    print("Ollama is accessible")

# Classify sentiment
result = analyzer.classify_sentiment("Great food and service!")
# Returns: {'sentiment': 'POSITIVE', 'confidence': 'high', 'reason': '...'}

# Extract issues
issues = analyzer.extract_issues("Food was cold and delivery was late")
# Returns: ['cold food', 'late delivery']
```

### ReviewAnalyzer

Main class for processing review datasets.

```python
analyzer = ReviewAnalyzer("reviews.xlsx")

# Load data
df = analyzer.load_data()

# Analyze sentiments
results = analyzer.analyze_sentiments(batch_size=10)

# Generate statistics
stats = analyzer.generate_summary_statistics()

# Detect trending issues
trending = analyzer.detect_trending_issues(min_frequency=2)

# Create visualizations
analyzer.create_visualizations("results.png")

# Save results
analyzer.save_results("results.xlsx")
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

# Talabat Product Operations Review Dashboard

This is a Flask web dashboard for customer sentiment analysis and Zomato operational metrics, ready for production hosting.

## Features
- Customer sentiment analysis with trending issues and suggestions
- Zomato customer experience and platform behavior metrics
- Fast dashboard loads with built-in caching (Flask-Caching)
- Downloadable Excel results

## Quick Start

1. **Install dependencies**

```bash
pip install -r flask_requirements.txt
```

2. **Run in Production with Gunicorn**

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

- `-w 4`: Number of worker processes (adjust as needed)
- `-b 0.0.0.0:5000`: Binds to all interfaces on port 5000

3. **(Optional) Use a Reverse Proxy**
- For production, use Nginx or Apache as a reverse proxy for SSL, static files, etc.

## Caching
- The dashboard is cached for 10 minutes for fast repeated loads.
- To manually clear the cache, visit: `http://<your-server>:5000/clear_cache`

## Customization
- Edit `app.py` to change analysis logic or data sources.
- Edit `templates/index.html` for UI changes.

## Excel Data
- Place your review and Zomato data Excel files in the project root as needed.

---

For any issues, please raise an issue or contact the maintainer.

## Version Control (Git)

To initialize a git repository and push your project to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository. 