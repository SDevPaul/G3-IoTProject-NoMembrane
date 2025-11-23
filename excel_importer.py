#!/usr/bin/env python3
"""
Simple Excel Importer Interface
Easy-to-use interface for importing Excel files with encryption
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try to import the processor
try:
    from excel_health_processor import SecureExcelHealthProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️  excel_health_processor.py not found in directory")
    PROCESSOR_AVAILABLE = False

def create_sample_excel():
    """Create a sample Excel file for demonstration."""
    print("\n📝 Creating sample Excel file for demonstration...")
    
    # Generate sample data
    np.random.seed(42)
    n_records = 200
    
    # Create sample data
    data = {
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_records)],
        'steps': np.random.poisson(500, n_records),
        'distance': np.random.normal(0.5, 0.2, n_records),
        'pace': np.random.normal(1.2, 0.3, n_records),
        'user': ['User1'] * (n_records//2) + ['User2'] * (n_records//2),
        'activity': np.random.choice(['walking', 'running', 'stationary'], n_records)
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_excel('sample_health_data.xlsx', index=False)
    
    print("✅ Sample Excel file created: sample_health_data.xlsx")
    return 'sample_health_data.xlsx'

def main():
    """Simple interface for Excel import with encryption."""
    print("=" * 60)
    print("📊 SECURE EXCEL HEALTH DATA IMPORTER")
    print("🔒 With AES-256 Encryption")
    print("=" * 60)
    
    if not PROCESSOR_AVAILABLE:
        print("\n❌ ERROR: Required files not found!")
        print("Make sure these files are in the same directory:")
        print("   - secure_crypto_manager.py")
        print("   - excel_health_processor.py (document 4)")
        print("   - excel_importer.py (this file)")
        sys.exit(1)
    
    # Initialize processor with encryption
    processor = SecureExcelHealthProcessor(
        master_password="SecureMasterPassword123"
    )
    
    # Check for Excel files
    excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
    
    if not excel_files:
        print("📝 No Excel files found in current directory")
        print("\n💡 Creating sample data for demonstration...")
        excel_file = create_sample_excel()
        excel_files = [excel_file]
    
    print(f"\n📁 Found {len(excel_files)} Excel file(s):")
    for i, file in enumerate(excel_files, 1):
        print(f"   {i}. {file}")
    
    # Process first Excel file
    excel_file = excel_files[0]
    print(f"\n🔄 Processing: {excel_file}")
    print("🔒 Data will be encrypted with AES-256-GCM")
    
    # Import data with encryption
    user_id = "User_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if processor.import_excel_data(
        excel_file, 
        user_id=user_id,
        password="user_secure_password"
    ):
        print("✅ Excel data imported and encrypted successfully")
        
        # Generate encrypted report
        report_file = processor.generate_report(encrypted=True)
        print(f"✅ Encrypted report generated: {report_file}")
        
        # Export encrypted results
        results_file = processor.export_results(encrypted=True)
        print(f"✅ Encrypted results exported: {results_file}")
        
        print("\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print("=" * 60)
        print("🔐 All data encrypted with AES-256-GCM")
        print("📁 Files created:")
        print(f"   - encrypted_data_{user_id}.json")
        print(f"   - {report_file}")
        print(f"   - {results_file}")
        print("\n🔒 Your health data is secure!")
        print("\n💡 To decrypt data, you need:")
        print(f"   - User ID: {user_id}")
        print("   - Password: user_secure_password")
        print("   - Master Password: SecureMasterPassword123")
        
        # Display some statistics (from unencrypted memory)
        print("\n📊 Data Statistics:")
        print(f"   Total Records: {len(processor.data)}")
        print(f"   Total Steps: {int(processor.data['step_count'].sum()):,}")
        print(f"   Date Range: {processor.data['timestamp'].min()} to {processor.data['timestamp'].max()}")
        print(f"   Activities: {processor.data['activity'].value_counts().to_dict()}")
        
    else:
        print("❌ Failed to import Excel data")
        print("💡 Make sure your Excel file has columns like:")
        print("   - timestamp/date/time")
        print("   - steps/step_count")
        print("   - distance")
        print("   - pace/speed")
        print("   - user/name/id")
        print("   - activity/type")

if __name__ == "__main__":
    main()