#!/usr/bin/env python3
"""
NFL Roster Data Download Script
Downloads roster data from nflverse-data GitHub releases for years 2015-2025
and saves them as CSV files locally.
"""

import os
import pandas as pd
import requests
from pathlib import Path
import time
from datetime import datetime

class NFLRosterDownloader:
    def __init__(self, base_dir="data/nflfastr_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URL for nflverse-data releases
        self.base_url = "https://github.com/nflverse/nflverse-data/releases/download/rosters"
        
        # Years to download (roster data typically goes back to 1999)
        self.years = list(range(2015, 2026))  # 2015-2025
        
        print(f"🏈 NFL Roster Data Downloader")
        print(f"📁 Data will be saved to: {self.base_dir.absolute()}")
        print(f"📅 Years to download: {self.years}")
    
    def download_roster_file(self, year):
        """Download roster data for a specific year"""
        
        # Different URL patterns to try
        url_patterns = [
            f"{self.base_url}/roster_{year}.csv",
            f"{self.base_url}/roster_{year}.parquet",
            f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.csv",
            f"https://github.com/nflverse/nflfastR-data/raw/master/data/roster_{year}.csv.gz"
        ]
        
        local_path = self.base_dir / f"rosters_{year}.csv"
        
        # Skip if file already exists and is valid
        if local_path.exists() and local_path.stat().st_size > 1000:
            try:
                df_test = pd.read_csv(local_path, nrows=1)
                print(f"   ✅ File already exists and is valid: {local_path.name}")
                return True
            except:
                print(f"   ⚠️  Existing file appears corrupted, re-downloading...")
        
        print(f"📥 Downloading {year} roster data...")
        
        for i, url in enumerate(url_patterns):
            try:
                print(f"   Trying URL {i+1}: {url}")
                
                # Handle different file types
                if url.endswith('.parquet'):
                    df = pd.read_parquet(url)
                elif url.endswith('.csv.gz'):
                    df = pd.read_csv(url, compression='gzip')
                else:
                    df = pd.read_csv(url)
                
                # Save to local CSV
                df.to_csv(local_path, index=False)
                
                print(f"   ✅ Downloaded successfully!")
                print(f"   📊 Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
                print(f"   💾 Saved to: {local_path}")
                
                return True
                
            except requests.exceptions.HTTPError as e:
                if "404" in str(e):
                    print(f"   ❌ File not found (404) - trying next URL...")
                    continue
                else:
                    print(f"   ❌ HTTP Error: {e}")
                    continue
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        print(f"   ❌ Failed to download {year} roster data from all URLs")
        return False
    
    def download_all_rosters(self):
        """Download roster data for all years"""
        
        print("\n" + "="*50)
        print("👥 DOWNLOADING ALL ROSTER DATA")
        print("="*50)
        
        success_count = 0
        failed_years = []
        
        for year in self.years:
            print(f"\n📅 Processing {year}...")
            
            if self.download_roster_file(year):
                success_count += 1
            else:
                failed_years.append(year)
            
            # Be nice to the server
            time.sleep(0.5)
        
        print(f"\n📊 Download Summary:")
        print(f"   ✅ Successful: {success_count}/{len(self.years)} files")
        
        if failed_years:
            print(f"   ❌ Failed years: {failed_years}")
        
        return success_count, failed_years
    
    def try_alternative_sources(self):
        """Try alternative data sources if main URLs fail"""
        
        print("\n" + "="*50)
        print("🔄 TRYING ALTERNATIVE SOURCES")
        print("="*50)
        
        # Try Lee Sharpe's nfldata repository (known to have rosters)
        lee_sharpe_url = "https://raw.githubusercontent.com/leesharpe/nfldata/master/data/rosters.csv"
        
        try:
            print("📥 Trying Lee Sharpe's nfldata repository...")
            df = pd.read_csv(lee_sharpe_url)
            
            print(f"   ✅ Found combined roster data!")
            print(f"   📊 Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Check what years are available
            if 'season' in df.columns:
                available_years = sorted(df['season'].unique())
                print(f"   📅 Available years: {min(available_years)}-{max(available_years)}")
                
                # Save individual year files
                for year in self.years:
                    if year in available_years:
                        year_df = df[df['season'] == year]
                        if len(year_df) > 0:
                            local_path = self.base_dir / f"rosters_{year}.csv"
                            year_df.to_csv(local_path, index=False)
                            print(f"   ✅ Saved {year}: {len(year_df):,} players")
                
                return True
            
        except Exception as e:
            print(f"   ❌ Failed to load from alternative source: {e}")
        
        return False
    
    def validate_downloads(self):
        """Validate all downloaded roster files"""
        
        print("\n" + "="*50)
        print("🔍 VALIDATING ROSTER FILES")
        print("="*50)
        
        total_size_mb = 0
        valid_files = 0
        
        for year in self.years:
            file_path = self.base_dir / f"rosters_{year}.csv"
            
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 / 1024
                total_size_mb += size_mb
                
                try:
                    df = pd.read_csv(file_path)
                    valid_files += 1
                    
                    # Show key info
                    teams = df['team'].nunique() if 'team' in df.columns else 'N/A'
                    players = len(df)
                    
                    print(f"   ✅ {year}: {size_mb:.1f} MB, {players:,} players, {teams} teams")
                    
                except Exception as e:
                    print(f"   ❌ {year}: {size_mb:.1f} MB, but CSV is invalid: {e}")
            else:
                print(f"   ❌ {year}: Missing")
        
        print(f"\n📈 Validation Summary:")
        print(f"   ✅ Valid files: {valid_files}/{len(self.years)}")
        print(f"   💾 Total size: {total_size_mb:.1f} MB")
        print(f"   📁 Location: {self.base_dir.absolute()}")
        
        return valid_files
    
    def show_sample_data(self):
        """Show sample data from the most recent roster file"""
        
        print("\n" + "="*50)
        print("👀 SAMPLE ROSTER DATA")
        print("="*50)
        
        # Find the most recent valid file
        for year in reversed(self.years):
            file_path = self.base_dir / f"rosters_{year}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"\n📊 Sample from {year} roster data:")
                    print(f"   📏 Total players: {len(df):,}")
                    print(f"   📋 Columns: {list(df.columns)}")
                    
                    # Show sample players
                    if len(df) > 0:
                        print(f"\n   🎯 Sample players:")
                        # Try different possible column names
                        name_cols = ['player_name', 'full_name', 'name', 'player_display_name']
                        team_cols = ['team', 'team_abbr', 'posteam']
                        pos_cols = ['position', 'pos']
                        
                        name_col = next((col for col in name_cols if col in df.columns), None)
                        team_col = next((col for col in team_cols if col in df.columns), None)
                        pos_col = next((col for col in pos_cols if col in df.columns), None)
                        
                        sample_cols = [col for col in [name_col, team_col, pos_col, 'jersey_number'] if col and col in df.columns]
                        
                        for i, (_, row) in enumerate(df.head(3).iterrows()):
                            sample_data = {col: row.get(col, 'N/A') for col in sample_cols}
                            print(f"     {i+1}. {sample_data}")
                    
                    # Show position breakdown if available
                    pos_col = next((col for col in ['position', 'pos'] if col in df.columns), None)
                    if pos_col:
                        print(f"\n   📈 Position breakdown:")
                        pos_counts = df[pos_col].value_counts().head(8)
                        for pos, count in pos_counts.items():
                            print(f"     {pos}: {count}")
                    
                    break
                    
                except Exception as e:
                    print(f"   ❌ Error reading {year} data: {e}")
                    continue
    
    def run(self):
        """Run the complete download and validation process"""
        
        print("🚀 Starting NFL roster data download...")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Download all roster files
        success_count, failed_years = self.download_all_rosters()
        
        # If many files failed, try alternative sources
        if len(failed_years) > len(self.years) // 2:
            print(f"\n⚠️  Many files failed ({len(failed_years)}/{len(self.years)}), trying alternative sources...")
            if self.try_alternative_sources():
                success_count = len(self.years) - len(failed_years)
        
        # Validate downloads
        valid_files = self.validate_downloads()
        
        # Show sample data
        self.show_sample_data()
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*50)
        print("🎉 DOWNLOAD COMPLETE!")
        print("="*50)
        print(f"⏱️  Total time: {duration:.1f} seconds")
        print(f"✅ Successfully downloaded: {success_count}/{len(self.years)} files")
        print(f"✅ Valid CSV files: {valid_files}/{len(self.years)} files")
        
        if failed_years:
            print(f"❌ Could not download: {failed_years}")
        
        print(f"📁 Files saved to: {self.base_dir.absolute()}")
        
        if valid_files > 0:
            print(f"\n🚀 Next Steps:")
            print(f"1. Use these CSV files in your analysis")
            print(f"2. Run your coaching analysis: python scripts/analysis/coaching_analysis.py")
        else:
            print(f"\n⚠️  No valid roster files were downloaded. Check internet connection and try again.")
        
        return valid_files > 0

def main():
    """Main function"""
    
    print("🏈 NFL Roster Data Download Tool")
    print("="*50)
    
    # Initialize and run downloader
    downloader = NFLRosterDownloader()
    success = downloader.run()
    
    if success:
        print("\n🎉 Roster data downloaded successfully!")
    else:
        print("\n⚠️  Failed to download roster data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()