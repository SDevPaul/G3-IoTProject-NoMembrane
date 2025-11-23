#!/usr/bin/env python3
"""
Run Data Visualization System
Simple script to process Excel data and generate visualizations
"""

import os
import json
import webbrowser
from data_visualization_processor import DataVisualizationProcessor

def main():
    """
    Main function to run the visualization system.
    """
    print("📊 IoT Health Data Visualization System")
    print("=" * 50)
    
    processor = DataVisualizationProcessor()
    
    # Check for Excel files
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
    
    if excel_files:
        print(f"📁 Found Excel files: {excel_files}")
        excel_file = excel_files[0]
        
        if processor.import_excel_data(excel_file):
            print("✅ Data imported successfully!")
            
            # Generate visualizations
            visualizations = processor.generate_visualizations()
            
            # Create summary report
            processor.create_summary_report()
            
            # Export processed data
            processor.export_processed_data()
            
            print("\n📊 Files generated:")
            print("- visualization_data.json")
            print("- data_analysis_report.json") 
            print("- processed_health_data.csv")
            
            # Open dashboard
            print("\n🌐 Opening visualization dashboard...")
            dashboard_path = os.path.abspath('data_visualization_dashboard.html')
            webbrowser.open(f'file://{dashboard_path}')
            
        else:
            print("❌ Failed to import Excel data.")
    else:
        print("📝 No Excel files found.")
        print("Please add an Excel file (.xlsx, .xls, .csv) to the directory and run again.")
        print("\nExample Excel format:")
        print("Date,Steps,Distance_km,Speed_kmh,Activity_Type,User_ID")
        print("2024-01-15 10:30:00,847,0.6,5.2,Walking,User_1")

if __name__ == "__main__":
    main()
