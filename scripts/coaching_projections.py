#!/usr/bin/env python3
"""
NFL Coaching Projections - Enhanced Z-Score Version
Creates projections based purely on historical play caller data
Features era-adjusted pace using z-score normalization
"""

import pandas as pd
import numpy as np
import os
import glob

def load_all_pbp_data() -> pd.DataFrame:
    """Load all play-by-play data from local CSV files (2015-2024)"""
    print("📊 Loading play-by-play data from local CSV files...")
    
    # Try different possible paths
    possible_paths = [
        '../../data/nflfastr_data/',  # From scripts/analysis/
        'data/nflfastr_data/',        # From project root
        '../data/nflfastr_data/',     # From scripts/
        './data/nflfastr_data/'       # Current directory
    ]
    
    pbp_files = []
    for path in possible_paths:
        # Look for custom pbp files first
        custom_files = glob.glob(f'{path}pbp_*_custom.csv')
        if custom_files:
            pbp_files = custom_files
            print(f"✅ Found custom pbp files in {path}")
            break
        
        # Look for regular pbp files
        regular_files = glob.glob(f'{path}pbp_*.csv')
        if regular_files:
            pbp_files = regular_files
            print(f"✅ Found regular pbp files in {path}")
            break
    
    if not pbp_files:
        print("❌ No pbp files found")
        return pd.DataFrame()
    
    print(f"Found {len(pbp_files)} pbp files")
    
    all_seasons = []
    
    for file_path in pbp_files:
        try:
            # Extract year from filename
            filename = os.path.basename(file_path)
            if '_custom.csv' in filename:
                year = int(filename.split('_')[1])  # pbp_2020_custom.csv
            else:
                year = int(filename.split('_')[1].split('.')[0])  # pbp_2020.csv
            
            # Only load 2015-2024
            if 2015 <= year <= 2024:
                print(f"  Loading {year}...")
                df = pd.read_csv(file_path)
                
                # Add season column if missing
                if 'season' not in df.columns:
                    df['season'] = year
                
                # Keep only necessary columns
                required_cols = ['season', 'posteam', 'play_type', 'game_id', 
                                'score_differential', 'qtr', 'wp', 'week', 'game_type']
                available_cols = [col for col in required_cols if col in df.columns]
                
                if 'posteam' in df.columns and 'play_type' in df.columns:
                    df_subset = df[available_cols].copy()
                    
                    # Filter to regular season only
                    if 'week' in df_subset.columns:
                        max_week = 17 if year >= 2021 else 16
                        df_subset = df_subset[df_subset['week'] <= max_week]
                        print(f"    Loaded {len(df_subset):,} regular season plays")
                    
                    all_seasons.append(df_subset)
                
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            continue
    
    if not all_seasons:
        print("❌ No valid pbp data loaded")
        return pd.DataFrame()
    
    # Combine all seasons
    combined_df = pd.concat(all_seasons, ignore_index=True)
    print(f"✅ Loaded {len(combined_df):,} total plays from {len(all_seasons)} seasons")
    
    return combined_df

def load_coaching_data() -> pd.DataFrame:
    """Load coaching data from CSV"""
    print("👨‍💼 Loading coaching data...")
    
    possible_files = [
        '../../data/playcaller_database.csv',
        'data/playcaller_database.csv',
        '../data/playcaller_database.csv',
        './data/playcaller_database.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Loaded coaching data: {len(df)} records")
            return df
    
    print("❌ Could not find playcaller_database.csv")
    return pd.DataFrame()

def filter_neutral_game_script(df: pd.DataFrame) -> pd.DataFrame:
    """Filter plays to neutral game script situations"""
    print("🎯 Filtering to neutral game script...")
    
    original_count = len(df)
    
    # Only offensive plays
    df = df[df['play_type'].isin(['pass', 'run'])].copy()
    
    # More generous neutral game script filters to get closer to 54% target
    if 'score_differential' in df.columns:
        df = df[(df['score_differential'] >= -10) & (df['score_differential'] <= 10)]  # Expanded from -7/+7
    
    if 'qtr' in df.columns:
        df = df[df['qtr'] <= 3]  # Keep this - not 4th quarter
    
    # Remove or relax win probability filter since it might be too restrictive
    # if 'wp' in df.columns:
    #     df = df[(df['wp'].isna()) | ((df['wp'] >= 0.2) & (df['wp'] <= 0.8))]
    
    filtered_count = len(df)
    print(f"✅ Kept {filtered_count:,} neutral plays ({filtered_count/original_count*100:.1f}% of total)")
    
    # Debug: Check if we're closer to 54%
    target_percentage = 54.0
    actual_percentage = (filtered_count/original_count*100)
    print(f"🎯 Target: {target_percentage}%, Actual: {actual_percentage:.1f}%, Difference: {actual_percentage - target_percentage:.1f}%")
    
    return df

def normalize_team_name(team):
    """Normalize team names to handle abbreviation differences"""
    if pd.isna(team):
        return team
    
    team = str(team).strip().upper()
    
    # Common team fixes
    team_fixes = {
        'LAR': 'LA',  # Coaching DB has LAR, play-by-play has LA
        'WSH': 'WAS'  # Sometimes Washington is WSH vs WAS
    }
    
    return team_fixes.get(team, team)

def normalize_play_caller_name(name):
    """Normalize play caller names to handle common variations"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Common name fixes
    name_fixes = {
        'Kevin OConnell': 'Kevin O\'Connell',
        'Kevin Oconnell': 'Kevin O\'Connell',
        'Kevin O Connell': 'Kevin O\'Connell',
    }
    
    return name_fixes.get(name, name)

def calculate_league_pace_by_year(seasons_df):
    """Calculate league average and standard deviation for neutral pace by year"""
    print("📊 Calculating yearly league pace baselines...")
    
    yearly_stats = {}
    
    for year in range(2015, 2025):
        year_data = seasons_df[seasons_df['season'] == year]
        
        if len(year_data) > 0:
            # Only include coordinators with meaningful sample sizes for that year
            reliable_data = year_data[year_data['neutral_plays'] >= 200]
            
            if len(reliable_data) >= 8:  # Need at least 8 teams for reliable league stats
                league_avg = reliable_data['neutral_pace'].mean()
                league_std = reliable_data['neutral_pace'].std()
                
                yearly_stats[year] = {
                    'avg': league_avg,
                    'std': league_std,
                    'teams': len(reliable_data)
                }
                
                print(f"  {year}: {league_avg:.1f} ± {league_std:.1f} pace ({len(reliable_data)} teams)")
            else:
                print(f"  {year}: Insufficient data ({len(reliable_data)} teams)")
    
    return yearly_stats

def calculate_coordinator_z_scores(seasons_df, yearly_stats):
    """Calculate z-scores for each coordinator season"""
    print("🧮 Calculating coordinator z-scores...")
    
    # Add z-score column
    seasons_df['pace_z_score'] = np.nan
    
    for idx, row in seasons_df.iterrows():
        year = row['season']
        pace = row['neutral_pace']
        
        if year in yearly_stats and not np.isnan(pace):
            year_stats = yearly_stats[year]
            z_score = (pace - year_stats['avg']) / year_stats['std']
            seasons_df.at[idx, 'pace_z_score'] = z_score
    
    return seasons_df

def analyze_play_caller_history():
    """Enhanced analysis with z-score pace calculations"""
    print("🔍 ANALYZING PLAY CALLER HISTORY (ENHANCED Z-SCORE)")
    print("=" * 55)
    
    # Load data
    pbp_df = load_all_pbp_data()
    coaching_df = load_coaching_data()
    
    if pbp_df.empty or coaching_df.empty:
        print("❌ Missing required data")
        return {}, {}
    
    # Normalize play caller names and team names
    coaching_df['play_caller_name'] = coaching_df['play_caller_name'].apply(normalize_play_caller_name)
    coaching_df['team'] = coaching_df['team'].apply(normalize_team_name)
    
    # Merge pbp with coaching data
    print("🔗 Merging play-by-play with coaching data...")
    merged_df = pbp_df.merge(
        coaching_df[['team', 'season', 'play_caller_name']], 
        left_on=['posteam', 'season'], 
        right_on=['team', 'season'], 
        how='inner'
    )
    
    print(f"✅ Merged data: {len(merged_df):,} plays with coaching info")
    
    # Debug: Check what teams we have data for
    print(f"🔍 DEBUG: Teams in merged data: {sorted(merged_df['team'].unique())}")
    
    # Filter to neutral game script
    neutral_df = filter_neutral_game_script(merged_df.copy())
    
    # Calculate play caller tendencies
    print("📈 Calculating play caller tendencies...")
    
    play_caller_seasons = []
    
    # Group by play caller, season, and team
    for (play_caller, season, team), group in merged_df.groupby(['play_caller_name', 'season', 'team']):
        
        # Calculate games
        if 'game_id' in group.columns:
            actual_games = group['game_id'].nunique()
            expected_games = 17 if season >= 2021 else 16
            # Use expected games if we're close (accounts for missing data)
            total_games = expected_games if actual_games >= expected_games - 1 else actual_games
        else:
            total_games = 17 if season >= 2021 else 16
        
        # Get neutral script plays for this caller/season/team
        neutral_group = neutral_df[
            (neutral_df['play_caller_name'] == play_caller) & 
            (neutral_df['season'] == season) & 
            (neutral_df['team'] == team)
        ]
        
        neutral_plays = len(neutral_group)
        neutral_pass_plays = len(neutral_group[neutral_group['play_type'] == 'pass'])
        
        # Only keep if we have decent sample size
        if neutral_plays >= 100:  # Minimum neutral plays for reliable data
            neutral_pass_rate = neutral_pass_plays / neutral_plays
            neutral_pace = neutral_plays / total_games  # Neutral script plays per game
            
            play_caller_seasons.append({
                'play_caller': play_caller,
                'season': season,
                'team': team,
                'total_games': total_games,
                'neutral_plays': neutral_plays,
                'neutral_pass_plays': neutral_pass_plays,
                'neutral_pass_rate': neutral_pass_rate,
                'neutral_pace': neutral_pace
            })
    
    if not play_caller_seasons:
        print("❌ No play caller seasons with sufficient data")
        return {}, {}
    
    seasons_df = pd.DataFrame(play_caller_seasons)
    print(f"📊 Analyzed {len(seasons_df)} play caller seasons")
    
    # NEW: Enhanced pace calculations with z-scores
    yearly_stats = calculate_league_pace_by_year(seasons_df)
    seasons_df = calculate_coordinator_z_scores(seasons_df, yearly_stats)
    
    # Calculate current league average for projection (use recent years)
    current_league_avg = 33.5  # Fallback
    recent_years = [2024, 2023, 2022]
    for year in recent_years:
        if year in yearly_stats:
            current_league_avg = yearly_stats[year]['avg']
            print(f"📈 Using {year} as baseline: {current_league_avg:.1f} pace")
            break
    
    # Enhanced pace profiles using z-scores
    enhanced_pace_profiles = {}
    
    for caller, group in seasons_df.groupby('play_caller'):
        # Filter out seasons without z-scores
        valid_seasons = group.dropna(subset=['pace_z_score'])
        
        if len(valid_seasons) == 0:
            continue
            
        # Sort by season (most recent first)
        valid_seasons = valid_seasons.sort_values('season', ascending=False)
        
        total_neutral_plays = valid_seasons['neutral_plays'].sum()
        
        # Skip if insufficient total data
        if total_neutral_plays < 300:
            continue
        
        # Calculate exponential weights (more recent = higher weight)
        years_back = np.arange(len(valid_seasons))
        weights = 0.85 ** years_back  # Each year back gets 0.85x weight
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate weighted average z-score
        weighted_z_score = np.average(valid_seasons['pace_z_score'].values, weights=weights)
        
        # Calculate confidence metrics
        seasons_count = len(valid_seasons)
        z_score_std = valid_seasons['pace_z_score'].std()
        z_score_consistency = 1.0 / (1.0 + z_score_std) if z_score_std > 0 else 1.0
        
        # Enhanced confidence calculation
        if seasons_count >= 4 and total_neutral_plays >= 1500 and z_score_consistency > 0.5:
            confidence = "HIGH"
            regression_factor = 0.1  # Minimal regression for high confidence
        elif seasons_count >= 3 and total_neutral_plays >= 800 and z_score_consistency > 0.3:
            confidence = "MEDIUM" 
            regression_factor = 0.25  # Light regression
        elif seasons_count >= 2 and total_neutral_plays >= 500:
            confidence = "LOW"
            regression_factor = 0.4   # Moderate regression
        else:
            confidence = "VERY_LOW"
            regression_factor = 0.6   # Heavy regression
        
        # Apply regression toward league average (z-score of 0)
        regressed_z_score = weighted_z_score * (1 - regression_factor)
        
        # Convert back to projected pace using most recent year's std dev
        recent_std = yearly_stats[2024]['std'] if 2024 in yearly_stats else 2.5
        projected_pace = current_league_avg + (regressed_z_score * recent_std)
        
        # Store enhanced profile
        enhanced_pace_profiles[caller] = {
            'seasons': seasons_count,
            'total_neutral_plays': int(total_neutral_plays),
            'weighted_z_score': weighted_z_score,
            'regressed_z_score': regressed_z_score,
            'projected_pace': projected_pace,
            'z_score_consistency': z_score_consistency,
            'confidence': confidence,
            'regression_factor': regression_factor,
            'season_z_scores': valid_seasons[['season', 'pace_z_score', 'neutral_pace']].to_dict('records')
        }
        
        # Debug output for key coordinators
        if any(name in caller for name in ['Kingsbury', 'Reid', 'McDaniel', 'McVay', 'Payton']):
            print(f"\n🔍 {caller}:")
            print(f"  Seasons: {seasons_count}, Total plays: {total_neutral_plays}")
            print(f"  Raw weighted z-score: {weighted_z_score:.2f}")
            print(f"  Regressed z-score: {regressed_z_score:.2f}")
            print(f"  Projected pace: {projected_pace:.1f}")
            print(f"  Confidence: {confidence}")
            for season_data in enhanced_pace_profiles[caller]['season_z_scores'][-3:]:  # Show last 3 seasons
                print(f"    {season_data['season']}: {season_data['neutral_pace']:.1f} pace (z={season_data['pace_z_score']:.2f})")
    
    print(f"✅ Created enhanced pace profiles for {len(enhanced_pace_profiles)} coordinators")
    
    # Original pass rate profiles (unchanged)
    pass_profiles = {}
    for caller, group in seasons_df.groupby('play_caller'):
        total_neutral_plays = group['neutral_plays'].sum()
        if total_neutral_plays < 300:
            continue
            
        neutral_pass_rate = group['neutral_pass_plays'].sum() / total_neutral_plays
        seasons = len(group)
        confidence = "HIGH" if seasons >= 4 and total_neutral_plays >= 1500 else "MEDIUM" if seasons >= 2 and total_neutral_plays >= 600 else "LOW"
        
        recent_seasons = group.nlargest(3, 'season')
        recent_neutral = recent_seasons['neutral_plays'].sum()
        recent_trend = None
        if recent_neutral >= 400:
            recent_trend = recent_seasons['neutral_pass_plays'].sum() / recent_neutral
        
        pass_profiles[caller] = {
            'seasons': seasons,
            'total_neutral_plays': int(total_neutral_plays),
            'neutral_pass_rate': neutral_pass_rate,
            'recent_trend': recent_trend,
            'confidence': confidence
        }
    
    print(f"✅ Created pass profiles for {len(pass_profiles)} coordinators")
    
    return pass_profiles, enhanced_pace_profiles

def create_projections():
    """Create 2025 projections with enhanced z-score pace calculations"""
    print(f"\n🎯 CREATING 2025 PROJECTIONS (ENHANCED Z-SCORE)")
    print("=" * 50)
    
    # Get enhanced profiles
    pass_profiles, enhanced_pace_profiles = analyze_play_caller_history()
    
    if not pass_profiles and not enhanced_pace_profiles:
        print("❌ No profiles available")
        return pd.DataFrame()
    
    # Load 2025 coaching assignments
    coaching_df = load_coaching_data()
    if coaching_df.empty:
        print("❌ No coaching data available")
        return pd.DataFrame()
    
    coaches_2025 = coaching_df[coaching_df['season'] == 2025].copy()
    coaches_2025['play_caller_name'] = coaches_2025['play_caller_name'].apply(normalize_play_caller_name)
    
    print(f"👨‍💼 Found {len(coaches_2025)} teams for 2025")
    
    # Calculate league averages for fallback
    if pass_profiles:
        league_neutral_pass_rate = np.mean([p['neutral_pass_rate'] for p in pass_profiles.values()])
    else:
        league_neutral_pass_rate = 0.57
    
    if enhanced_pace_profiles:
        current_league_avg = np.mean([p['projected_pace'] for p in enhanced_pace_profiles.values()])
    else:
        current_league_avg = 33.5
    
    print(f"📊 League averages: {league_neutral_pass_rate:.1%} pass rate, {current_league_avg:.1f} neutral pace")
    print(f"📊 Enhanced pace profiles: {len(enhanced_pace_profiles)} coordinators")
    print(f"📊 Pass rate profiles: {len(pass_profiles)} coordinators")
    
    print(f"\n🎯 Creating projections for {len(coaches_2025)} teams...")
    print("=" * 50)
    
    # Create projections
    projections = []
    
    for _, coach_row in coaches_2025.iterrows():
        team = coach_row['team']
        play_caller = coach_row['play_caller_name']
        
        # Get pass rate (from existing logic)
        if play_caller in pass_profiles:
            profile = pass_profiles[play_caller]
            if profile['recent_trend'] is not None:
                pass_rate = profile['recent_trend']
                pass_data_source = f"Recent trend ({profile['seasons']} seasons)"
            else:
                pass_rate = profile['neutral_pass_rate']
                pass_data_source = f"Career average ({profile['seasons']} seasons)"
            pass_confidence = profile['confidence']
        else:
            pass_rate = league_neutral_pass_rate
            pass_data_source = "League average (no historical data)"
            pass_confidence = "LOW"
        
        # Get enhanced pace
        if play_caller in enhanced_pace_profiles:
            pace_profile = enhanced_pace_profiles[play_caller]
            projected_pace = pace_profile['projected_pace']
            pace_confidence = pace_profile['confidence']
            
            if pace_profile['regression_factor'] > 0.3:
                pace_data_source = f"Z-score regressed ({pace_profile['seasons']} seasons, z={pace_profile['regressed_z_score']:.2f})"
            else:
                pace_data_source = f"Z-score weighted ({pace_profile['seasons']} seasons, z={pace_profile['regressed_z_score']:.2f})"
        else:
            projected_pace = current_league_avg
            pace_confidence = "LOW"
            pace_data_source = "League average (no historical data)"
        
        # Combined confidence (take lower of pass/pace confidence)
        confidence_ranking = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "VERY_LOW": 0}
        combined_confidence_score = min(confidence_ranking[pass_confidence], confidence_ranking[pace_confidence])
        combined_confidence = [k for k, v in confidence_ranking.items() if v == combined_confidence_score][0]
        
        # Ensure realistic bounds
        pass_rate = max(0.45, min(0.75, pass_rate))
        projected_pace = max(28.0, min(40.0, projected_pace))  # Slightly wider bounds for pace
        
        projections.append({
            'team': team,
            'play_caller': play_caller,
            'projected_pass_rate': round(pass_rate, 3),
            'projected_rush_rate': round(1.0 - pass_rate, 3),
            'projected_neutral_pace': round(projected_pace, 1),
            'projected_neutral_pass_plays': round(projected_pace * pass_rate, 1),
            'projected_neutral_rush_plays': round(projected_pace * (1.0 - pass_rate), 1),
            'confidence': combined_confidence,
            'data_source': f"Pass: {pass_data_source} | Pace: {pace_data_source}"
        })
        
        # Debug output
        print(f"  {team} ({play_caller}) [{combined_confidence}]:")
        if "Z-score" in pace_data_source:
            z_match = pace_data_source.split('z=')[1].split(')')[0]
            if pace_profile['regression_factor'] > 0.3:
                print(f"    Pace: {projected_pace:.1f} plays/game (z={z_match}) 📉 REGRESSED")
            else:
                print(f"    Pace: {projected_pace:.1f} plays/game (z={z_match}) ⚡ ERA-ADJUSTED")
        else:
            print(f"    Pace: {projected_pace:.1f} plays/game ({pace_data_source})")
        print(f"    Pass: {pass_rate:.1%} ({pass_data_source})")
    
    df_proj = pd.DataFrame(projections)
    print(f"✅ Created {len(df_proj)} enhanced projections")
    
    return df_proj

def show_results(df_proj):
    """Enhanced results display"""
    
    print(f"\n📊 2025 NFL PROJECTIONS (ENHANCED Z-SCORE)")
    print("=" * 45)
    
    # Pace leaders with z-score context
    df_pace = df_proj.sort_values('projected_neutral_pace', ascending=False)
    print(f"\n⚡ TOP 10 PACE LEADERS (Era-Adjusted):")
    for i, (_, row) in enumerate(df_pace.head(10).iterrows()):
        conf_icon = "🔥" if row['confidence'] == "HIGH" else "⚡" if row['confidence'] == "MEDIUM" else "❓"
        z_score_info = ""
        if "z=" in row['data_source']:
            z_match = row['data_source'].split('z=')[1].split(')')[0]
            z_score_info = f" (z={z_match})"
        print(f"   {i+1:2d}. {row['team']}: {row['play_caller']:<20} {row['projected_neutral_pace']:.1f}{z_score_info} {conf_icon}")
    
    # Pass leaders
    df_pass = df_proj.sort_values('projected_pass_rate', ascending=False)
    print(f"\n🎯 TOP 5 PASS-HEAVY TEAMS:")
    for i, (_, row) in enumerate(df_pass.head(5).iterrows()):
        conf_icon = "🔥" if row['confidence'] == "HIGH" else "⚡" if row['confidence'] == "MEDIUM" else "❓"
        print(f"   {i+1}. {row['team']}: {row['play_caller']:<20} {row['projected_pass_rate']:.1%} {conf_icon}")
    
    # Run-heavy teams
    print(f"\n🏃 MOST RUN-HEAVY TEAMS:")
    for i, (_, row) in enumerate(df_pass.tail(5).iterrows()):
        conf_icon = "🔥" if row['confidence'] == "HIGH" else "⚡" if row['confidence'] == "MEDIUM" else "❓"
        print(f"   {row['team']}: {row['play_caller']:<20} {row['projected_pass_rate']:.1%} {conf_icon}")
    
    # Show specific coordinators of interest
    print(f"\n🔍 KEY COORDINATORS:")
    key_names = ['Kingsbury', 'Reid', 'McDaniel', 'McVay', 'Payton']
    for name in key_names:
        matching_row = df_proj[df_proj['play_caller'].str.contains(name, na=False)]
        if not matching_row.empty:
            row = matching_row.iloc[0]
            pace_rank = (df_pace['projected_neutral_pace'] > row['projected_neutral_pace']).sum() + 1
            print(f"   {row['play_caller']:<18} ({row['team']}): {row['projected_neutral_pace']:.1f} pace (#{pace_rank}), {row['projected_pass_rate']:.1%} pass")
    
    # Enhanced statistics
    print(f"\n📈 ENHANCED LEAGUE STATISTICS:")
    print(f"   Pass Rate: {df_proj['projected_pass_rate'].mean():.1%}")
    print(f"   Neutral Pace: {df_proj['projected_neutral_pace'].mean():.1f} plays/game")
    print(f"   Pace Range: {df_proj['projected_neutral_pace'].min():.1f} - {df_proj['projected_neutral_pace'].max():.1f}")
    print(f"   Pace Spread: {df_proj['projected_neutral_pace'].max() - df_proj['projected_neutral_pace'].min():.1f} plays/game")
    
    # Confidence breakdown
    print(f"\n🎯 CONFIDENCE BREAKDOWN:")
    conf_counts = df_proj['confidence'].value_counts()
    for conf, count in conf_counts.items():
        print(f"   {conf}: {count} teams")

def save_projections(df_proj):
    """Save enhanced projections"""
    
    print(f"\n💾 SAVING ENHANCED PROJECTIONS")
    print("=" * 30)
    
    try:
        # Try different output paths
        possible_output_dirs = [
            '../../data/projections_output',
            'data/projections_output', 
            '../data/projections_output',
            './data/projections_output'
        ]
        
        output_dir = None
        for dir_path in possible_output_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                output_dir = dir_path
                break
            except:
                continue
        
        if output_dir is None:
            output_dir = '.'
        
        filename = os.path.join(output_dir, 'nfl_projections_2025_enhanced_zscore.csv')
        
        # Select columns for export
        export_df = df_proj[[
            'team', 'play_caller', 'projected_pass_rate', 'projected_rush_rate',
            'projected_neutral_pace', 'projected_neutral_pass_plays', 'projected_neutral_rush_plays', 
            'confidence', 'data_source'
        ]].copy()
        
        export_df.to_csv(filename, index=False)
        print(f"✅ Saved to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False

def main():
    """Enhanced main function"""
    
    print("🏈 NFL COACHING PROJECTIONS - ENHANCED Z-SCORE VERSION")
    print("=" * 60)
    print("Features:")
    print("✅ Z-score normalized pace (era-adjusted)")
    print("✅ Exponential weighting (recent seasons matter more)")
    print("✅ Reduced regression for consistent performers")
    print("✅ Enhanced confidence calculations")
    
    # Create enhanced projections
    df_proj = create_projections()
    
    if not df_proj.empty:
        # Show results
        show_results(df_proj)
        
        # Save results
        if save_projections(df_proj):
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Created enhanced z-score projections")
            print(f"🎯 Pace leaders should now reflect era-adjusted performance!")
        else:
            print(f"\n⚠️  Projections created but not saved")
    else:
        print(f"\n❌ Failed to create projections")

if __name__ == "__main__":
    main()