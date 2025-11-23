#!/usr/bin/env python3
"""
Create sample Excel file for demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_excel():
    """Create a sample Excel file with health data"""
    
    print("📊 Creating sample Excel file with health data...")
    
    # Generate sample data
    np.random.seed(42)
    n_records = 200
    
    # Create date range
    start_date = datetime.now() - timedelta(days=7)
    dates = [start_date + timedelta(hours=i) for i in range(n_records)]
    
    # Generate realistic health data
    data = []
    activities = ['Walking', 'Running', 'Stationary']
    
    for i, date in enumerate(dates):
        # Simulate daily patterns
        hour = date.hour
        
        if 6 <= hour <= 8 or 17 <= hour <= 19:  # Commute times
            activity = 'Walking'
            steps = np.random.poisson(800)
            distance = np.random.normal(0.6, 0.2)
            speed = np.random.normal(5, 1)
        elif 9 <= hour <= 17:  # Work hours
            activity = 'Stationary'
            steps = np.random.poisson(50)
            distance = np.random.normal(0.05, 0.02)
            speed = np.random.normal(0.5, 0.2)
        elif 19 <= hour <= 21:  # Exercise time
            activity = 'Running'
            steps = np.random.poisson(1500)
            distance = np.random.normal(1.2, 0.3)
            speed = np.random.normal(8, 2)
        else:  # Night time
            activity = 'Stationary'
            steps = np.random.poisson(20)
            distance = np.random.normal(0.01, 0.005)
            speed = np.random.normal(0.1, 0.05)
        
        # Ensure non-negative values
        steps = max(0, int(steps))
        distance = max(0, distance)
        speed = max(0, speed)
        
        data.append({
            'Date': date,
            'Steps': steps,
            'Distance_km': round(distance, 2),
            'Speed_kmh': round(speed, 1),
            'Activity_Type': activity,
            'User_ID': f'User_{(i % 3) + 1}',
            'Duration_minutes': np.random.randint(5, 120),
            'Heart_Rate': np.random.normal(70, 15),
            'Calories': np.random.poisson(50)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    excel_file = 'sample_health_data.xlsx'
    df.to_excel(excel_file, index=False)
    
    print(f"✅ Sample Excel file created: {excel_file}")
    print(f"📊 Records: {len(df)}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"🏃 Activity distribution:")
    print(df['Activity_Type'].value_counts().to_dict())
    
    return excel_file

if __name__ == "__main__":
    create_sample_excel()
