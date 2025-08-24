#!/usr/bin/env python3
"""
Final Working NFL FastR Data Download Script
Fixes compression and parquet issues
"""

import os
import pandas as pd
import requests
from pathlib import Path
import time
import gzip
import shutil
from datetime import datetime

class FinalNFLDownloader:
    def __init__(self, base_dir="data/nflfastr_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Correct URLs for nflverse-data
        self.base_url = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
        
        # Your requested years
        self.years = list(range(2015, 2025))
        
        # Your custom columns
        self.selected_columns = [
            # Core Game Data
            'game_id', 'season', 'week', 'game_date',
            'posteam', 'defteam', 'home_team', 'away_team',
            
            # Game Situation  
            'down', 'ydstogo', 'yardline_100', 'qtr',
            'score_differential', 'posteam_score', 'defteam_score',
            'wp', 'game_seconds_remaining',
            
            # Play Details
            'play_type', 'yards_gained',
            'passer_player_name', 'rusher_player_name', 'receiver_player_name',
            
            # EPA & Efficiency
            'epa', 'wpa', 'success', 'cpoe',
            
            # Pass Details
            'air_yards', 'yards_after_catch',
            'complete_pass', 'incomplete_pass', 'interception',
            'pass_location', 'pass_length',
            
            # Rush Details
            'rush_direction',
            
            # Context
            'shotgun', 'no_huddle', 'play_clock',
            'penalty', 'touchdown', 'field_goal_result',
            
            # Drive Context
            'drive', 'drive_play_count', 'drive_time_of_possession',
            
            # Weather
            'temp', 'wind',
            
            # Vegas
            'vegas_wp', 'vegas_home_wp',
            
            # Additional
            'roof', 'surface', 'timeouts_remaining',
            'total_home_score', 'total_away_score'
        ]
        
        print(f"🏈 Final NFL Data Downloader (All Issues Fixed)")
        print(f"📁 Data directory: {self.base_dir.absolute()}")
        print(f"📅 Years: {self.years[0]}-{self.years[-1]}")
        print(f"📊 Columns: {len(self.selected_columns)}")
    
    def download_file(self, url, local_path):
        """Download a file with proper error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"      Progress: {progress:.1f}%", end='\r')
            
            print(f"\n   ✅ Downloaded: {downloaded / 1024 / 1024:.1f} MB")
            return True
            
        except Exception as e:
            print(f"\n   ❌ Download failed: {e}")
            return False
    
    def read_data_file(self, file_path, file_type):
        """Read data file handling different formats"""
        try:
            if file_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_type == 'csv_gz':
                # Properly handle gzipped CSV
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            elif file_type == 'csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unknown file type: {file_type}")
            
            print(f"   📊 Loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"   ❌ Read error: {e}")
            return None
    
    def download_and_process_year(self, year):
        """Download and process one year of data"""
        
        print(f"\n📅 Processing {year}...")
        
        final_path = self.base_dir / f"pbp_{year}_custom.csv"
        
        # Skip if already exists
        if final_path.exists() and final_path.stat().st_size > 50000:
            size_mb = final_path.stat().st_size / 1024 / 1024
            print(f"   ✅ Already exists: {size_mb:.1f} MB")
            return True
        
        # Try different file formats
        download_attempts = [
            ('parquet', f"{self.base_url}/play_by_play_{year}.parquet"),
            ('csv_gz', f"{self.base_url}/play_by_play_{year}.csv.gz"),
        ]
        
        for file_type, url in download_attempts:
            print(f"   🔍 Trying {file_type}: {url.split('/')[-1]}")
            
            temp_path = self.base_dir / f"temp_{year}.{file_type.split('_')[0]}"
            
            # Download
            if not self.download_file(url, temp_path):
                continue
            
            # Read
            df = self.read_data_file(temp_path, file_type)
            if df is None:
                if temp_path.exists():
                    os.remove(temp_path)
                continue
            
            # Process the data
            original_size = len(df)
            
            # Filter columns
            available_cols = [col for col in self.selected_columns if col in df.columns]
            missing_cols = [col for col in self.selected_columns if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing {len(missing_cols)} columns: {missing_cols[:3]}...")
            
            df_filtered = df[available_cols].copy()
            print(f"   🔽 Using {len(available_cols)} columns")
            
            # Apply filters
            print(f"   🎯 Applying filters...")
            
            # Pass/run plays only
            if 'play_type' in df_filtered.columns:
                before = len(df_filtered)
                df_filtered = df_filtered[df_filtered['play_type'].isin(['pass', 'run'])]
                print(f"      Play types: {len(df_filtered):,}/{before:,}")
            
            # Remove missing key data
            key_cols = ['posteam', 'play_type']
            available_key = [col for col in key_cols if col in df_filtered.columns]
            if available_key:
                before = len(df_filtered)
                df_filtered = df_filtered.dropna(subset=available_key)
                print(f"      Data quality: {len(df_filtered):,}/{before:,}")
            
            # Competitive games only
            if 'score_differential' in df_filtered.columns:
                before = len(df_filtered)
                df_filtered = df_filtered[
                    (df_filtered['score_differential'] >= -21) & 
                    (df_filtered['score_differential'] <= 21)
                ]
                print(f"      Score filter: {len(df_filtered):,}/{before:,}")
            
            # Remove garbage time
            if 'qtr' in df_filtered.columns and 'game_seconds_remaining' in df_filtered.columns:
                before = len(df_filtered)
                df_filtered = df_filtered[
                    (df_filtered['qtr'] <= 3) | 
                    ((df_filtered['qtr'] == 4) & (df_filtered['game_seconds_remaining'] > 180))
                ]
                print(f"      Time filter: {len(df_filtered):,}/{before:,}")
            
            # Save
            df_filtered.to_csv(final_path, index=False)
            
            # Cleanup
            if temp_path.exists():
                os.remove(temp_path)
            
            # Report
            final_size = len(df_filtered)
            file_size_mb = final_path.stat().st_size / 1024 / 1024
            
            print(f"   ✅ Success!")
            print(f"      Final: {final_size:,} plays ({final_size/original_size*100:.1f}% retention)")
            print(f"      Size: {file_size_mb:.1f} MB")
            
            return True
        
        print(f"   ❌ All download attempts failed for {year}")
        return False
    
    def download_all(self):
        """Download all years"""
        
        print(f"\n" + "="*70)
        print(f"📊 DOWNLOADING ALL NFL DATA")
        print(f"="*70)
        
        success_count = 0
        total_size = 0
        failed_years = []
        
        for i, year in enumerate(self.years):
            success = self.download_and_process_year(year)
            
            if success:
                success_count += 1
                file_path = self.base_dir / f"pbp_{year}_custom.csv"
                if file_path.exists():
                    total_size += file_path.stat().st_size / 1024 / 1024
            else:
                failed_years.append(year)
            
            # Progress update
            print(f"\n   📈 Progress: {i+1}/{len(self.years)} years processed")
            
            # Rest between downloads
            if i < len(self.years) - 1:
                print(f"   ⏳ Waiting 2 seconds...")
                time.sleep(2)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   ✅ Successful: {success_count}/{len(self.years)}")
        print(f"   📁 Total size: {total_size:.1f} MB")
        print(f"   🎯 Supabase limit: 500 MB")
        
        if failed_years:
            print(f"   ❌ Failed years: {failed_years}")
        
        status = "✅ Perfect!" if total_size < 400 else "⚠️ Close to limit" if total_size < 500 else "❌ Too large"
        print(f"   {status}")
        
        return success_count, total_size

def main():
    """Main function"""
    
    print("🏈 Final NFL Data Downloader")
    print("=" * 50)
    
    # Check dependencies
    missing = []
    
    try:
        import pandas as pd
    except ImportError:
        missing.append('pandas')
    
    try:
        import requests
    except ImportError:
        missing.append('requests')
    
    try:
        import pyarrow
    except ImportError:
        missing.append('pyarrow')
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print(f"Run: pip install {' '.join(missing)}")
        return
    
    print(f"✅ All packages available")
    
    # Initialize
    downloader = FinalNFLDownloader()
    
    # Confirm
    response = input(f"\n🤔 Download {len(downloader.years)} years of data? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Download
    start_time = time.time()
    success_count, total_size = downloader.download_all()
    duration = time.time() - start_time
    
    print(f"\n🎉 COMPLETE!")
    print(f"⏱️  Time: {duration/60:.1f} minutes")
    print(f"✅ Downloaded: {success_count} years")
    print(f"📊 Size: {total_size:.1f} MB")
    
    if success_count >= 8:
        print(f"\n🚀 Ready for next steps:")
        print(f"1. Load to Supabase")
        print(f"2. Create projections")
    else:
        print(f"\n🔧 May need troubleshooting")

if __name__ == "__main__":
    main()