
from flask import Flask, render_template, send_file, jsonify
import os
import pandas as pd
from datetime import datetime
from flask_caching import Cache
from functools import lru_cache
import threading

app = Flask(__name__)

# Configure Flask-Caching with longer timeout and better config
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache', 
    'CACHE_DEFAULT_TIMEOUT': 1800,  # 30 minutes instead of 10
    'CACHE_THRESHOLD': 500
})

# Global variables for caching data in memory
_zomato_data_cache = None
_zomato_data_timestamp = None
_cache_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_static_sentiment_data():
    """Cache static sentiment data in memory"""
    return {
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

def check_file_exists(file_path):
    """Quick file existence check"""
    return os.path.isfile(file_path)

def get_cached_zomato_data():
    """Get Zomato data with intelligent caching"""
    global _zomato_data_cache, _zomato_data_timestamp
    
    file_path = 'Zomato mock data.xlsx'
    
    # Check if file exists first (fast operation)
    if not check_file_exists(file_path):
        return {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }
    
    # Check file modification time
    try:
        file_mtime = os.path.getmtime(file_path)
        
        with _cache_lock:
            # Return cached data if file hasn't changed
            if (_zomato_data_cache is not None and 
                _zomato_data_timestamp is not None and 
                _zomato_data_timestamp >= file_mtime):
                return _zomato_data_cache
            
            # Load and cache new data
            _zomato_data_cache = analyze_zomato_data_optimized(file_path)
            _zomato_data_timestamp = file_mtime
            
        return _zomato_data_cache
        
    except Exception as e:
        print(f"Error getting Zomato data: {e}")
        return {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }

def analyze_zomato_data_optimized(file_path):
    """Optimized Zomato data analysis with vectorized operations"""
    try:
        # Read only necessary columns to reduce memory usage
        required_columns = ['order_delay', 'is_acquisition', 'is_successful', 'owner', 'platform', 'gmv_amount_lc']
        
        # Read Excel file (chunksize not supported for Excel)
        try:
            df = pd.read_excel(file_path, usecols=lambda x: x in required_columns)
        except:
            # Fallback to reading all columns if usecols fails
            df = pd.read_excel(file_path)
        
        zomato_data = {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }
        
        # Vectorized operations for better performance
        # 1. Customer experience metrics
        if 'order_delay' in df.columns:
            order_delays = df['order_delay'].dropna()
            if not order_delays.empty:
                avg_delay = float(order_delays.mean())
                percent_delayed = float((order_delays > 0).mean() * 100)
                zomato_data['customer_experience']['avg_delay'] = round(avg_delay, 2)
                zomato_data['customer_experience']['percent_delayed'] = round(percent_delayed, 2)
        
        # First-time Order Success Rate (vectorized)
        if 'is_acquisition' in df.columns and 'is_successful' in df.columns:
            first_time_mask = df['is_acquisition'] == True
            if first_time_mask.any():
                success_rate_new = float(df.loc[first_time_mask, 'is_successful'].mean() * 100)
                zomato_data['customer_experience']['first_time_success_rate'] = round(success_rate_new, 2)
        
        # Delay Comparison (optimized groupby)
        if 'is_acquisition' in df.columns and 'order_delay' in df.columns:
            delay_comparison = df.groupby('is_acquisition')['order_delay'].mean()
            zomato_data['customer_experience']['delay_comparison'] = {
                str(k): round(float(v), 2) for k, v in delay_comparison.items()
            }
        
        # 2. Failure & Root Cause Trends (vectorized)
        if 'is_successful' in df.columns:
            failure_rate = float((df['is_successful'] == False).mean() * 100)
            zomato_data['failure_trends']['overall_failure_rate'] = round(failure_rate, 2)
        
        # Failure by Owner (optimized)
        if 'is_successful' in df.columns and 'owner' in df.columns:
            failures = df[df['is_successful'] == False]
            if not failures.empty:
                failure_counts = failures['owner'].value_counts().head(10)  # Limit to top 10
                zomato_data['failure_trends']['failure_by_owner'] = {
                    str(k): int(v) for k, v in failure_counts.items()
                }
        
        # 3. Platform Behavior (optimized groupby)
        if 'platform' in df.columns and 'is_successful' in df.columns:
            platform_success = df.groupby('platform')['is_successful'].mean() * 100
            zomato_data['platform_behavior']['success_by_platform'] = {
                str(k): round(float(v), 2) for k, v in platform_success.items()
            }
        
        if 'platform' in df.columns and 'gmv_amount_lc' in df.columns:
            platform_gmv = df.groupby('platform')['gmv_amount_lc'].mean()
            zomato_data['platform_behavior']['avg_gmv_by_platform'] = {
                str(k): round(float(v), 2) for k, v in platform_gmv.items()
            }
        
        return zomato_data
        
    except Exception as e:
        print(f"Error analyzing Zomato data: {e}")
        return {
            'customer_experience': {},
            'failure_trends': {},
            'platform_behavior': {}
        }

@app.route("/clear_cache")
def clear_cache():
    """Clear all caches"""
    global _zomato_data_cache, _zomato_data_timestamp
    
    cache.clear()
    get_static_sentiment_data.cache_clear()
    
    with _cache_lock:
        _zomato_data_cache = None
        _zomato_data_timestamp = None
    
    return "All caches cleared!", 200

@app.route("/", methods=["GET"])
@cache.cached(timeout=1800)  # 30 minutes cache
def index():
    """Main dashboard route with optimized data loading"""
    # Get cached data efficiently
    sentiment_data = get_static_sentiment_data()
    zomato_data = get_cached_zomato_data()
    
    # Combine data
    combined_data = {
        'sentiment': sentiment_data,
        'zomato': zomato_data
    }
    
    return render_template('index.html', data=combined_data)

@app.route('/download_results')
@cache.cached(timeout=3600)  # Cache for 1 hour
def download_results():
    """Optimized file download with caching"""
    try:
        file_path = 'comprehensive_analysis_results.xlsx'
        if check_file_exists(file_path):
            return send_file(file_path, as_attachment=True, download_name='talabat_sentiment_analysis_results.xlsx')
        else:
            # Create sample file efficiently
            create_sample_excel_optimized()
            return send_file(file_path, as_attachment=True, download_name='talabat_sentiment_analysis_results.xlsx')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def create_sample_excel_optimized():
    """Optimized Excel file creation"""
    # Pre-define data structures for better performance
    sample_data = {
        'Review': ['Great service!', 'Poor delivery time', 'App works well'],
        'Sentiment': ['POSITIVE', 'NEGATIVE', 'POSITIVE'],
        'English_Reason': ['Good service mentioned', 'Delivery issues', 'App functionality praised'],
        'English_Issues': [['None'], ['Slow delivery'], ['None']],
        'Positive_Aspects': [['Good service'], ['None'], ['App works well']],
        'Customer_Suggestions': [['None'], ['Improve delivery'], ['None']]
    }
    
    # Get cached sentiment data for summary
    sentiment_stats = get_static_sentiment_data()
    
    summary_data = [
        ['Total Reviews Analyzed', sentiment_stats['total_reviews']],
        ['', ''],
        ['Sentiment Distribution', ''],
        ['NEGATIVE', f"{sentiment_stats['sentiment_distribution']['NEGATIVE']} reviews ({sentiment_stats['sentiment_percentages']['NEGATIVE']}%)"],
        ['POSITIVE', f"{sentiment_stats['sentiment_distribution']['POSITIVE']} reviews ({sentiment_stats['sentiment_percentages']['POSITIVE']}%)"],
        ['NEUTRAL', f"{sentiment_stats['sentiment_distribution']['NEUTRAL']} reviews ({sentiment_stats['sentiment_percentages']['NEUTRAL']}%)"],
        ['', ''],
        ['Top Issues (Most Repeated)', ''],
        *[[issue, f'{count} mentions'] for issue, count in sentiment_stats['top_issues']],
        ['', ''],
        ['Top Positive Aspects', ''],
        *[[aspect, f'{count} mentions'] for aspect, count in sentiment_stats['top_positive_aspects']],
        ['', ''],
        ['Top Customer Suggestions', ''],
        *[[suggestion, f'{count} mentions'] for suggestion, count in sentiment_stats['top_suggestions']]
    ]
    
    df = pd.DataFrame(sample_data)
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    # Use openpyxl engine for better performance
    with pd.ExcelWriter('comprehensive_analysis_results.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Reviews with Analysis', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

@app.route('/api/sentiment-data')
@cache.cached(timeout=3600)  # Cache API responses
def get_sentiment_data():
    """API endpoint for sentiment data with caching"""
    return jsonify(get_static_sentiment_data())

@app.route('/api/zomato-data')
@cache.cached(timeout=1800)  # Cache for 30 minutes
def get_zomato_data():
    """API endpoint for Zomato data with caching"""
    return jsonify(get_cached_zomato_data())

# Optimize Flask settings for production
if __name__ == '__main__':
    # Check if running in deployment
    is_deployment = os.getenv('REPLIT_DEPLOYMENT') == '1'
    
    app.run(
        debug=not is_deployment,  # Disable debug in production
        host='0.0.0.0', 
        port=5000,
        threaded=True,
        use_reloader=False
    )
