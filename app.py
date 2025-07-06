from flask import Flask, render_template, send_file, jsonify
import os
import pandas as pd
from datetime import datetime
from flask_caching import Cache

app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 600})

def analyze_zomato_data():
    """Analyze Zomato data and return metrics"""
    try:
        # Read the Excel file
        file_path = 'Zomato mock data.xlsx'
        df = pd.read_excel(file_path)
        
        zomato_data = {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }
        
        # 1. Customer experience metrics
        if 'order_delay' in df.columns:
            df['is_delayed'] = df['order_delay'] > 0
            avg_delay = df['order_delay'].mean()
            percent_delayed = df['is_delayed'].mean() * 100
            zomato_data['customer_experience']['avg_delay'] = round(avg_delay, 2)
            zomato_data['customer_experience']['percent_delayed'] = round(percent_delayed, 2)
        
        # First-time Order Success Rate
        if 'is_acquisition' in df.columns and 'is_successful' in df.columns:
            first_time = df[df['is_acquisition'] == True]
            if not first_time.empty:
                success_rate_new = first_time['is_successful'].mean() * 100
                zomato_data['customer_experience']['first_time_success_rate'] = round(success_rate_new, 2)
        
        # Delay Comparison (New vs Returning)
        if 'is_acquisition' in df.columns and 'order_delay' in df.columns:
            delay_comparison = df.groupby('is_acquisition')['order_delay'].mean()
            zomato_data['customer_experience']['delay_comparison'] = delay_comparison.to_dict()
        
        # 2. Failure & Root Cause Trends
        if 'is_successful' in df.columns:
            failure_rate = (df['is_successful'] == False).mean() * 100
            zomato_data['failure_trends']['overall_failure_rate'] = round(failure_rate, 2)
        
        # Failure by Owner
        if 'is_successful' in df.columns and 'owner' in df.columns:
            failure_by_owner = df[df['is_successful'] == False].groupby('owner').size().sort_values(ascending=False)
            zomato_data['failure_trends']['failure_by_owner'] = failure_by_owner.to_dict()
        
        # 3. Platform Behavior
        if 'platform' in df.columns and 'is_successful' in df.columns:
            success_by_platform = df.groupby('platform')['is_successful'].mean() * 100
            zomato_data['platform_behavior']['success_by_platform'] = success_by_platform.to_dict()
        
        if 'platform' in df.columns and 'gmv_amount_lc' in df.columns:
            gmv_by_platform = df.groupby('platform')['gmv_amount_lc'].mean()
            zomato_data['platform_behavior']['avg_gmv_by_platform'] = gmv_by_platform.to_dict()
        
        return zomato_data
        
    except Exception as e:
        print(f"Error analyzing Zomato data: {e}")
        return {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }

# Sample data for demonstration
SENTIMENT_DATA = {
    'total_reviews': 499,
    'sentiment_distribution': {
        'NEGATIVE': 261,
        'POSITIVE': 219,
        'NEUTRAL': 19
    },
    'sentiment_percentages': {
        'NEGATIVE': 52.3,
        'POSITIVE': 43.89,
        'NEUTRAL': 3.81
    },
    'top_issues': [
        ('High delivery fees', 45),
        ('Slow delivery time', 38),
        ('Poor customer service', 32),
        ('App crashes frequently', 28)
    ],
    'top_positive_aspects': [
        ('Good food quality', 42),
        ('Easy to use app', 35),
        ('Wide restaurant selection', 28)
    ],
    'top_suggestions': [
        ('Reduce delivery fees', 38),
        ('Improve app stability', 25),
        ('Better customer support', 22)
    ]
}

@app.route("/clear_cache")
def clear_cache():
    cache.clear()
    return "Cache cleared!", 200

@app.route("/", methods=["GET"])
@cache.cached()
def index():
    # Get Zomato analysis data
    zomato_data = analyze_zomato_data()
    
    # Combine sentiment and Zomato data
    combined_data = {
        'sentiment': SENTIMENT_DATA,
        'zomato': zomato_data
    }
    
    return render_template('index.html', data=combined_data)

@app.route('/download_results')
def download_results():
    try:
        # Check if the file exists
        file_path = 'comprehensive_analysis_results.xlsx'
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='talabat_sentiment_analysis_results.xlsx')
        else:
            # Create a sample file if it doesn't exist
            create_sample_excel()
            return send_file(file_path, as_attachment=True, download_name='talabat_sentiment_analysis_results.xlsx')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def create_sample_excel():
    """Create a sample Excel file for download"""
    sample_data = {
        'Review': ['Great service!', 'Poor delivery time', 'App works well'],
        'Sentiment': ['POSITIVE', 'NEGATIVE', 'POSITIVE'],
        'English_Reason': ['Good service mentioned', 'Delivery issues', 'App functionality praised'],
        'English_Issues': [['None'], ['Slow delivery'], ['None']],
        'Positive_Aspects': [['Good service'], ['None'], ['App works well']],
        'Customer_Suggestions': [['None'], ['Improve delivery'], ['None']]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create summary data
    summary_data = [
        ['Total Reviews Analyzed', 499],
        ['', ''],
        ['Sentiment Distribution', ''],
        ['NEGATIVE', '261 reviews (52.3%)'],
        ['POSITIVE', '219 reviews (43.89%)'],
        ['NEUTRAL', '19 reviews (3.81%)'],
        ['', ''],
        ['Top 4 Issues (Most Repeated)', ''],
        ['High delivery fees', '45 mentions'],
        ['Slow delivery time', '38 mentions'],
        ['Poor customer service', '32 mentions'],
        ['App crashes frequently', '28 mentions'],
        ['', ''],
        ['Top 3 Positive Aspects', ''],
        ['Good food quality', '42 mentions'],
        ['Easy to use app', '35 mentions'],
        ['Wide restaurant selection', '28 mentions'],
        ['', ''],
        ['Top 3 Customer Suggestions', ''],
        ['Reduce delivery fees', '38 mentions'],
        ['Improve app stability', '25 mentions'],
        ['Better customer support', '22 mentions']
    ]
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    with pd.ExcelWriter('comprehensive_analysis_results.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Reviews with Analysis', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

@app.route('/api/sentiment-data')
def get_sentiment_data():
    return jsonify(SENTIMENT_DATA)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 