import pandas as pd

# Read the Excel file
file_path = 'Zomato mock data.xlsx'
df = pd.read_excel(file_path)

# 1. Customer experience metrics
print('--- Customer Experience Metrics ---')
# a. Average Delay & % Delayed Orders
if 'order_delay' in df.columns:
    df['is_delayed'] = df['order_delay'] > 0
    avg_delay = df['order_delay'].mean()
    percent_delayed = df['is_delayed'].mean() * 100
    print(f'Average Delay (seconds): {avg_delay:.2f}')
    print(f'% Delayed Orders: {percent_delayed:.2f}%')
else:
    print('order_delay column not found.')

# b. First-time Order Success Rate
if 'is_acquisition' in df.columns and 'is_successful' in df.columns:
    first_time = df[df['is_acquisition'] == True]
    if not first_time.empty:
        success_rate_new = first_time['is_successful'].mean() * 100
        print(f'First-time Order Success Rate: {success_rate_new:.2f}%')
    else:
        print('No first-time orders found.')
else:
    print('is_acquisition or is_successful column not found.')

# c. Delay Comparison (New vs Returning)
if 'is_acquisition' in df.columns and 'order_delay' in df.columns:
    delay_comparison = df.groupby('is_acquisition')['order_delay'].mean()
    print('Delay Comparison (New vs Returning):')
    print(delay_comparison)
else:
    print('is_acquisition or order_delay column not found.')

# 2. Failure & Root Cause Trends
print('\n--- Failure & Root Cause Trends ---')
# a. Overall Failure Rate
if 'is_successful' in df.columns:
    failure_rate = (df['is_successful'] == False).mean() * 100
    print(f'Overall Failure Rate: {failure_rate:.2f}%')
else:
    print('is_successful column not found.')

# b. Failure by Owner
if 'is_successful' in df.columns and 'owner' in df.columns:
    failure_by_owner = df[df['is_successful'] == False].groupby('owner').size().sort_values(ascending=False)
    print('Failure by Owner:')
    print(failure_by_owner)
else:
    print('is_successful or owner column not found.')

# c. Failure Heatmap (Reason vs Subreason)
if 'is_successful' in df.columns and 'reason' in df.columns and 'sub_reason' in df.columns:
    failure_heatmap = df[df['is_successful'] == False].pivot_table(index='reason', columns='sub_reason', aggfunc='size', fill_value=0)
    print('Failure Heatmap (Reason vs Subreason):')
    print(failure_heatmap)
else:
    print('is_successful, reason, or sub_reason column not found.')

# 3. Platform Behavior
print('\n--- Platform Behavior ---')
# a. Success Rate by Platform
if 'platform' in df.columns and 'is_successful' in df.columns:
    success_by_platform = df.groupby('platform')['is_successful'].mean() * 100
    print('Success Rate by Platform:')
    print(success_by_platform)
else:
    print('platform or is_successful column not found.')

# b. Avg GMV by Platform
if 'platform' in df.columns and 'gmv_amount_lc' in df.columns:
    gmv_by_platform = df.groupby('platform')['gmv_amount_lc'].mean()
    print('Average GMV by Platform:')
    print(gmv_by_platform)
else:
    print('platform or gmv_amount_lc column not found.') 