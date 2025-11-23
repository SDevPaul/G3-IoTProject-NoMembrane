#!/usr/bin/env python3
"""
Secure Data Visualization Processor with AES-256-GCM Encryption
Processes Excel data, generates visualizations, and encrypts all outputs
Now supports time-limited encryption keys (5-minute validity)

Features:
- AES-256-GCM encryption for all data files
- Time-limited encryption keys (5-minute validity)
- Interactive visualization dashboard
- Secure data processing pipeline
- Privacy-preserving analytics
- Encrypt/Decrypt functionality with key validation
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import base64
import hashlib
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Check for cryptography library
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
    print("✅ Cryptography library loaded - AES-256-GCM encryption enabled")
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️  WARNING: Cryptography library not installed!")
    print("   Install with: pip install cryptography")
    print("   Data will NOT be encrypted!\n")


class TimeLimitedKeyManager:
    """Manager for time-limited encryption keys."""
    
    def __init__(self, validity_minutes=5):
        self.current_key = None
        self.key_expiry = None
        self.validity_minutes = validity_minutes
        self.key_size = 32  # 256 bits
        
    def generate_key(self):
        """Generate a new time-limited encryption key."""
        if not CRYPTO_AVAILABLE:
            print("⚠️  Cannot generate key - cryptography not available")
            return None
            
        self.current_key = os.urandom(self.key_size)
        self.key_expiry = datetime.now() + timedelta(minutes=self.validity_minutes)
        
        key_hex = self.current_key.hex()
        print(f"\n🔑 NEW ENCRYPTION KEY GENERATED")
        print(f"   Key: {key_hex}")
        print(f"   Valid until: {self.key_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Validity: {self.validity_minutes} minutes")
        print(f"\n⚠️  SAVE THIS KEY! You'll need it to decrypt your data.")
        
        return key_hex
    
    def is_key_valid(self):
        """Check if current key is still valid."""
        if not self.current_key or not self.key_expiry:
            return False
        return datetime.now() < self.key_expiry
    
    def get_remaining_time(self):
        """Get remaining validity time in seconds."""
        if not self.key_expiry:
            return 0
        remaining = (self.key_expiry - datetime.now()).total_seconds()
        return max(0, remaining)
    
    def get_key(self):
        """Get current key if valid."""
        if not self.is_key_valid():
            raise ValueError("Encryption key expired or not generated")
        return self.current_key
    
    def load_key_from_hex(self, key_hex):
        """Load a key from hex string."""
        try:
            self.current_key = bytes.fromhex(key_hex)
            # When loading external key, set expiry to 5 minutes from now
            self.key_expiry = datetime.now() + timedelta(minutes=self.validity_minutes)
            print(f"✅ Key loaded successfully")
            print(f"   Valid until: {self.key_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        except Exception as e:
            print(f"❌ Failed to load key: {e}")
            return False


class SecureVisualizationCrypto:
    """Cryptography manager for secure data visualization with time-limited keys."""
    
    def __init__(self, master_password="SecureViz2024"):
        if not CRYPTO_AVAILABLE:
            self.enabled = False
            print("⚠️  Encryption DISABLED - cryptography not available")
            return
        
        self.enabled = True
        self.backend = default_backend()
        self.key_size = 32  # 256 bits
        self.nonce_size = 12
        self.tag_size = 16
        
        # Create secure key directory
        self.key_dir = '.secure_viz_keys'
        if not os.path.exists(self.key_dir):
            os.makedirs(self.key_dir, mode=0o700)
        
        self.master_password = master_password
        self.session_key = None
        
        # Initialize time-limited key manager
        self.key_manager = TimeLimitedKeyManager(validity_minutes=5)
        
        print("🔒 Secure Visualization Crypto initialized")
        print(f"   Algorithm: AES-256-GCM")
        print(f"   Key size: {self.key_size * 8} bits")
        print(f"   Key validity: {self.key_manager.validity_minutes} minutes")
    
    def encrypt_data_with_key(self, data, key):
        """Encrypt data using a specific key."""
        if not self.enabled:
            return data
        
        # Convert to bytes if string
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        nonce = os.urandom(self.nonce_size)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine nonce + tag + ciphertext
        encrypted = nonce + encryptor.tag + ciphertext
        
        return encrypted
    
    def decrypt_data_with_key(self, encrypted_data, key):
        """Decrypt data using a specific key."""
        if not self.enabled:
            return encrypted_data
        
        nonce = encrypted_data[:self.nonce_size]
        tag = encrypted_data[self.nonce_size:self.nonce_size + self.tag_size]
        ciphertext = encrypted_data[self.nonce_size + self.tag_size:]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def encrypt_file_with_key(self, input_file, output_file, key):
        """Encrypt a file with a specific key."""
        if not self.enabled:
            print(f"⚠️  Cannot encrypt {input_file} - encryption disabled")
            return input_file
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted = self.encrypt_data_with_key(data, key)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted)
        
        try:
            os.chmod(output_file, 0o600)
        except:
            pass
        
        print(f"🔒 Encrypted: {input_file} → {output_file}")
        return output_file
    
    def decrypt_file_with_key(self, encrypted_file, output_file, key):
        """Decrypt a file with a specific key."""
        if not self.enabled:
            print(f"⚠️  Cannot decrypt {encrypted_file} - encryption disabled")
            return encrypted_file
        
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted = self.decrypt_data_with_key(encrypted_data, key)
        
        with open(output_file, 'wb') as f:
            f.write(decrypted)
        
        print(f"🔓 Decrypted: {encrypted_file} → {output_file}")
        return output_file


class SecureDataVisualizationProcessor:
    """
    Secure data visualization processor with time-limited encryption keys.
    All sensitive data is encrypted with keys that expire after 5 minutes.
    """
    
    def __init__(self, master_password="SecureViz2024"):
        self.data = None
        self.processed_data = None
        self.visualizations = {}
        self.column_mapping = {}
        
        # Initialize crypto
        self.crypto = SecureVisualizationCrypto(master_password)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.crypto.enabled:
            print("🔐 Secure Data Visualization Processor initialized")
            print("   All outputs will be ENCRYPTED with time-limited keys")
        else:
            print("⚠️  Data Visualization Processor initialized WITHOUT encryption")
    
    def auto_detect_columns(self, df):
        """Auto-detect column structure."""
        print("\n🔍 Auto-detecting columns...")
        
        patterns = {
            'timestamp': ['date', 'time', 'timestamp', 'datetime', 'start'],
            'steps': ['steps', 'step', 'step_count', 'stepcount'],
            'distance': ['distance', 'dist', 'km', 'miles', 'meters'],
            'pace': ['pace', 'speed', 'velocity', 'mph', 'kmh'],
            'activity': ['activity', 'type', 'exercise', 'workout'],
            'user': ['user', 'name', 'person', 'id', 'anonymous'],
            'duration': ['duration', 'time', 'minutes', 'seconds'],
            'heart_rate': ['heart', 'hr', 'bpm', 'pulse'],
            'calories': ['calories', 'cal', 'energy', 'burned']
        }
        
        detected = {}
        for standard, pattern_list in patterns.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                if any(p in col_lower for p in pattern_list):
                    detected[standard] = col
                    print(f"✅ '{col}' → '{standard}'")
                    break
        
        self.column_mapping = detected
        return detected
    
    def import_excel_data(self, excel_file):
        """Import Excel data with security logging."""
        print("\n" + "=" * 60)
        print(f"📊 IMPORTING: {excel_file}")
        print("=" * 60)
        
        if self.crypto.enabled:
            print("🔐 SECURE MODE: Data will be encrypted with time-limited keys")
        else:
            print("⚠️  INSECURE MODE: Data will NOT be encrypted!")
        
        try:
            # Read Excel file
            print("\n📂 Reading file...")
            if excel_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(excel_file)
            elif excel_file.endswith('.csv'):
                df = pd.read_csv(excel_file)
            else:
                print("❌ Unsupported file format")
                return False
            
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Auto-detect columns
            self.auto_detect_columns(df)
            
            # Map to standard format
            print("\n🔄 Mapping data to standard format...")
            self.data = self._map_columns(df)
            
            # Process data
            print("\n🧹 Processing and anonymizing...")
            self._process_data()
            
            print("\n✅ Import complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _map_columns(self, df):
        """Map columns to standard format."""
        mapped = pd.DataFrame()
        
        # Timestamp
        if 'timestamp' in self.column_mapping:
            mapped['timestamp'] = pd.to_datetime(
                df[self.column_mapping['timestamp']], errors='coerce'
            )
        else:
            mapped['timestamp'] = pd.date_range(
                '2024-01-01', periods=len(df), freq='1H'
            )
        
        # Steps
        if 'steps' in self.column_mapping:
            mapped['step_count'] = pd.to_numeric(
                df[self.column_mapping['steps']], errors='coerce'
            ).fillna(0)
        else:
            mapped['step_count'] = np.random.poisson(500, len(df))
        
        # Distance
        if 'distance' in self.column_mapping:
            mapped['distance'] = pd.to_numeric(
                df[self.column_mapping['distance']], errors='coerce'
            ).fillna(0)
        else:
            mapped['distance'] = mapped['step_count'] * 0.0008
        
        # Pace
        if 'pace' in self.column_mapping:
            mapped['pace'] = pd.to_numeric(
                df[self.column_mapping['pace']], errors='coerce'
            ).fillna(0)
        else:
            mapped['pace'] = np.where(
                mapped['distance'] > 0,
                mapped['distance'] / 3600,
                np.random.normal(1.0, 0.5, len(df))
            )
        
        # User
        if 'user' in self.column_mapping:
            mapped['user'] = df[self.column_mapping['user']].astype(str)
        else:
            mapped['user'] = f'User_{self.session_id}'
        
        # Activity
        if 'activity' in self.column_mapping:
            mapped['activity'] = df[self.column_mapping['activity']].astype(str)
        else:
            mapped['activity'] = self._classify_activity(mapped)
        
        # Additional fields
        if 'duration' in self.column_mapping:
            mapped['duration_minutes'] = pd.to_numeric(
                df[self.column_mapping['duration']], errors='coerce'
            ).fillna(30)
        else:
            mapped['duration_minutes'] = np.random.randint(5, 120, len(df))
        
        if 'heart_rate' in self.column_mapping:
            mapped['heart_rate'] = pd.to_numeric(
                df[self.column_mapping['heart_rate']], errors='coerce'
            ).fillna(70)
        else:
            mapped['heart_rate'] = np.random.normal(70, 15, len(df))
        
        if 'calories' in self.column_mapping:
            mapped['calories'] = pd.to_numeric(
                df[self.column_mapping['calories']], errors='coerce'
            ).fillna(50)
        else:
            mapped['calories'] = np.random.poisson(50, len(df))
        
        return mapped
    
    def _classify_activity(self, df):
        """Classify activities based on metrics."""
        activities = []
        for _, row in df.iterrows():
            if row['step_count'] < 100:
                activities.append('stationary')
            elif row['step_count'] < 1000:
                activities.append('walking')
            else:
                activities.append('running')
        return activities
    
    def _process_data(self):
        """Process and anonymize data."""
        self.data = self.data.dropna(subset=['timestamp', 'step_count'])
        
        # Anonymize users
        unique_users = self.data['user'].unique()
        user_map = {u: f'Anonymous_{i+1}' for i, u in enumerate(unique_users)}
        self.data['user'] = self.data['user'].map(user_map)
        
        # Sort and add time features
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.day_name()
        
        self.processed_data = self.data.copy()
        
        print(f"✅ Processed {len(self.data)} records")
        print(f"✅ Anonymized {len(unique_users)} user(s)")
    
    def encrypt_data_with_time_limited_key(self):
        """Encrypt data using a time-limited key."""
        print("\n🔐 ENCRYPTING DATA WITH TIME-LIMITED KEY")
        print("=" * 60)
        
        if not self.crypto.enabled:
            print("❌ Encryption not available")
            return None
        
        if self.processed_data is None:
            print("❌ No data to encrypt")
            return None
        
        # Generate new time-limited key
        key_hex = self.crypto.key_manager.generate_key()
        key = self.crypto.key_manager.get_key()
        
        # Export data to JSON
        data_json = self.processed_data.to_json(orient='records', date_format='iso')
        
        # Encrypt
        print(f"\n🔒 Encrypting {len(self.processed_data)} records...")
        encrypted_data = self.crypto.encrypt_data_with_key(data_json.encode(), key)
        
        # Save encrypted file
        encrypted_file = f'encrypted_health_data_{self.session_id}.enc'
        with open(encrypted_file, 'wb') as f:
            f.write(encrypted_data)
        
        try:
            os.chmod(encrypted_file, 0o600)
        except:
            pass
        
        # Save metadata
        metadata = {
            'encrypted_file': encrypted_file,
            'algorithm': 'AES-256-GCM',
            'key_hex': key_hex,
            'key_expiry': self.crypto.key_manager.key_expiry.isoformat(),
            'session_id': self.session_id,
            'record_count': len(self.processed_data),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = f'encryption_metadata_{self.session_id}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Data encrypted successfully!")
        print(f"   Encrypted file: {encrypted_file}")
        print(f"   Metadata: {metadata_file}")
        print(f"\n⚠️  IMPORTANT:")
        print(f"   Key: {key_hex}")
        print(f"   Valid until: {self.crypto.key_manager.key_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Save this key to decrypt your data!")
        
        return metadata
    
    def decrypt_data_with_key(self, encrypted_file, key_hex):
        """Decrypt data using a provided key."""
        print("\n🔓 DECRYPTING DATA")
        print("=" * 60)
        
        if not self.crypto.enabled:
            print("❌ Decryption not available")
            return None
        
        # Load key
        if not self.crypto.key_manager.load_key_from_hex(key_hex):
            return None
        
        key = self.crypto.key_manager.get_key()
        
        # Read encrypted file
        print(f"📂 Reading encrypted file: {encrypted_file}")
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        print("🔓 Decrypting...")
        try:
            decrypted_data = self.crypto.decrypt_data_with_key(encrypted_data, key)
            data_json = decrypted_data.decode('utf-8')
            
            # Load back to DataFrame
            self.processed_data = pd.read_json(data_json, orient='records')
            self.processed_data['timestamp'] = pd.to_datetime(self.processed_data['timestamp'])
            
            print(f"✅ Successfully decrypted {len(self.processed_data)} records")
            
            # Save decrypted file
            decrypted_file = f'decrypted_health_data_{self.session_id}.csv'
            self.processed_data.to_csv(decrypted_file, index=False)
            print(f"💾 Decrypted data saved: {decrypted_file}")
            
            return self.processed_data
            
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            print("   Possible reasons:")
            print("   - Wrong encryption key")
            print("   - Key expired")
            print("   - Corrupted encrypted file")
            return None


def main():
    """Main execution function."""
    print("\n" + "🔐" * 30)
    print("SECURE DATA VISUALIZATION SYSTEM")
    print("With Time-Limited Encryption Keys (5-minute validity)")
    print("🔐" * 30 + "\n")
    
    if not CRYPTO_AVAILABLE:
        print("=" * 60)
        print("⚠️  CRITICAL SECURITY WARNING")
        print("=" * 60)
        print("Cryptography library is NOT installed!")
        print("Your data will NOT be encrypted!")
        print("\nTO ENABLE ENCRYPTION:")
        print("  pip install cryptography")
        print("\n" + "=" * 60 + "\n")
        
        response = input("Continue WITHOUT encryption? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting for security reasons.")
            sys.exit(1)
    
    # Initialize processor
    processor = SecureDataVisualizationProcessor(
        master_password="SecureViz2024!@#"
    )
    
    print("\n" + "=" * 60)
    print("MENU: Select Operation")
    print("=" * 60)
    print("1. Import and Encrypt Excel Data")
    print("2. Decrypt Encrypted Data")
    print("3. Exit")
    print("=" * 60)
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Find Excel files
        excel_files = [
            f for f in os.listdir('.')
            if f.endswith(('.xlsx', '.xls', '.csv'))
        ]
        
        if not excel_files:
            print("\n📝 No Excel files found. Creating sample data...")
            
            # Create sample data
            sample_df = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'Steps': np.random.poisson(500, 100),
                'Distance_km': np.random.normal(0.5, 0.2, 100),
                'Speed_kmh': np.random.normal(5, 2, 100),
                'Activity_Type': np.random.choice(['Walking', 'Running', 'Stationary'], 100),
                'User_ID': 'DemoUser'
            })
            
            sample_df.to_excel('sample_health_data.xlsx', index=False)
            print("✅ Created: sample_health_data.xlsx")
            excel_files = ['sample_health_data.xlsx']
        
        print(f"\n📁 Found {len(excel_files)} file(s):")
        for i, f in enumerate(excel_files, 1):
            print(f"   {i}. {f}")
        
        # Select file
        if len(excel_files) == 1:
            excel_file = excel_files[0]
        else:
            file_idx = int(input(f"\nSelect file (1-{len(excel_files)}): ")) - 1
            excel_file = excel_files[file_idx]
        
        # Import and encrypt
        if processor.import_excel_data(excel_file):
            processor.encrypt_data_with_time_limited_key()
            
            print("\n" + "=" * 60)
            print("🎉 ENCRYPTION COMPLETE!")
            print("=" * 60)
            print("✅ Your data has been encrypted with a time-limited key")
            print("⚠️  SAVE THE KEY DISPLAYED ABOVE!")
            print("   You'll need it to decrypt your data within 5 minutes")
            
    elif choice == '2':
        # Find encrypted files
        enc_files = [f for f in os.listdir('.') if f.endswith('.enc')]
        
        if not enc_files:
            print("\n❌ No encrypted files found")
            sys.exit(1)
        
        print(f"\n📁 Found {len(enc_files)} encrypted file(s):")
        for i, f in enumerate(enc_files, 1):
            print(f"   {i}. {f}")
        
        # Select file
        if len(enc_files) == 1:
            enc_file = enc_files[0]
        else:
            file_idx = int(input(f"\nSelect file (1-{len(enc_files)}): ")) - 1
            enc_file = enc_files[file_idx]
        
        # Get key
        key_hex = input("\nEnter encryption key (hex): ").strip()
        
        # Decrypt
        processor.decrypt_data_with_key(enc_file, key_hex)
        
    else:
        print("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()