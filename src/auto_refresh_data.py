#!/usr/bin/env python3
"""
Automated Yelp Data Refresh System
==================================

This script provides multiple ways to keep your Yelp data up-to-date:

1. Manual refresh with timestamps
2. Incremental updates (only fetch new/changed data)
3. Full refresh with backup
4. Scheduled refresh via cron
5. Data validation and integrity checks

Usage:
    # Manual full refresh
    python src/auto_refresh_data.py --mode full
    
    # Incremental update (recommended for daily use)
    python src/auto_refresh_data.py --mode incremental
    
    # Check data freshness
    python src/auto_refresh_data.py --mode check
    
    # Setup cron job for daily updates
    python src/auto_refresh_data.py --setup-cron
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from yelp_fetch_reviews import main as fetch_yelp_data
from prepare_business_metrics import main as prepare_metrics
from build_rag_index import main as build_rag

# Load .env file if it exists (for local development)
# In GitHub Actions, environment variables are set directly
load_dotenv()

# Verify required environment variables
if not os.getenv("YELP_API_KEY"):
    print("❌ Missing YELP_API_KEY in environment variables or .env file")
    print("   For GitHub Actions: Set YELP_API_KEY as a GitHub Secret")
    print("   For local use: Create .env file with YELP_API_KEY=your_key")
    sys.exit(1)

class YelpDataRefresher:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.backup_dir = self.data_dir / "backups"
        
        # Ensure directories exist
        self.backup_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Metadata file to track data freshness
        self.metadata_file = self.processed_dir / "data_metadata.json"
        
    def load_metadata(self) -> Dict:
        """Load data metadata including last update time and data stats."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "last_full_refresh": None,
            "last_incremental_update": None,
            "total_businesses": 0,
            "total_reviews": 0,
            "data_version": "1.0.0",
            "refresh_history": []
        }
    
    def save_metadata(self, metadata: Dict):
        """Save updated metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a timestamped backup of current data."""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Copy processed files
        for file in self.processed_dir.glob("*.csv"):
            shutil.copy2(file, backup_path / file.name)
        
        # Copy metadata
        if self.metadata_file.exists():
            shutil.copy2(self.metadata_file, backup_path / "data_metadata.json")
        
        print(f"✅ Backup created: {backup_path}")
        return str(backup_path)
    
    def check_data_freshness(self) -> Dict:
        """Check how fresh the current data is."""
        metadata = self.load_metadata()
        
        if not metadata.get("last_full_refresh"):
            return {
                "status": "no_data",
                "message": "No data found. Run full refresh first.",
                "days_old": None
            }
        
        last_refresh = datetime.fromisoformat(metadata["last_full_refresh"])
        days_old = (datetime.now() - last_refresh).days
        
        if days_old <= 1:
            status = "fresh"
            message = f"Data is fresh (updated {days_old} day(s) ago)"
        elif days_old <= 7:
            status = "stale"
            message = f"Data is getting stale (updated {days_old} days ago)"
        else:
            status = "outdated"
            message = f"Data is outdated (updated {days_old} days ago)"
        
        return {
            "status": status,
            "message": message,
            "days_old": days_old,
            "last_refresh": metadata["last_full_refresh"],
            "total_businesses": metadata.get("total_businesses", 0),
            "total_reviews": metadata.get("total_reviews", 0)
        }
    
    def validate_data_integrity(self) -> Dict:
        """Validate that the data files exist and are not corrupted."""
        required_files = [
            "businesses_clean.csv",
            "businesses_ranked.csv", 
            "business_metrics.csv"
        ]
        
        results = {
            "valid": True,
            "missing_files": [],
            "corrupted_files": [],
            "file_stats": {}
        }
        
        for file_name in required_files:
            file_path = self.processed_dir / file_name
            
            if not file_path.exists():
                results["missing_files"].append(file_name)
                results["valid"] = False
                continue
            
            try:
                # Try to read the file
                df = pd.read_csv(file_path)
                results["file_stats"][file_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": file_path.stat().st_size / (1024 * 1024)
                }
                
                # Basic validation
                if len(df) == 0:
                    results["corrupted_files"].append(f"{file_name} (empty)")
                    results["valid"] = False
                    
            except Exception as e:
                results["corrupted_files"].append(f"{file_name} ({str(e)})")
                results["valid"] = False
        
        return results
    
    def run_full_refresh(self, create_backup: bool = True) -> Dict:
        """Run a complete data refresh."""
        print("🔄 Starting full data refresh...")
        
        if create_backup:
            self.create_backup()
        
        start_time = datetime.now()
        
        try:
            # Step 1: Fetch fresh Yelp data
            print("📡 Fetching fresh Yelp data...")
            os.chdir(self.project_root)
            
            # Run the yelp fetch script
            result = subprocess.run([
                sys.executable, "src/yelp_fetch_reviews.py"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Yelp fetch failed: {result.stderr}")
            
            # Step 2: Process the data
            print("⚙️ Processing business metrics...")
            prepare_metrics()
            
            # Step 3: Build RAG index
            print("🔍 Building RAG index...")
            build_rag()
            
            # Step 4: Update metadata
            metadata = self.load_metadata()
            metadata["last_full_refresh"] = datetime.now().isoformat()
            metadata["data_version"] = f"1.{int(time.time())}"
            
            # Count businesses and reviews
            businesses_df = pd.read_csv(self.processed_dir / "businesses_clean.csv")
            metadata["total_businesses"] = len(businesses_df)
            metadata["total_reviews"] = businesses_df["review_count"].sum()
            
            # Add to refresh history
            refresh_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "full_refresh",
                "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "businesses_count": len(businesses_df),
                "reviews_count": businesses_df["review_count"].sum()
            }
            metadata["refresh_history"].append(refresh_record)
            
            # Keep only last 10 refresh records
            metadata["refresh_history"] = metadata["refresh_history"][-10:]
            
            self.save_metadata(metadata)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            print(f"✅ Full refresh completed in {duration:.1f} minutes")
            
            return {
                "success": True,
                "duration_minutes": duration,
                "businesses_count": len(businesses_df),
                "reviews_count": businesses_df["review_count"].sum()
            }
            
        except Exception as e:
            print(f"❌ Full refresh failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_incremental_update(self) -> Dict:
        """Run an incremental update (only fetch new/changed data)."""
        print("🔄 Starting incremental data update...")
        
        # For now, incremental update is the same as full refresh
        # In a production system, you'd implement smart diffing
        # based on business IDs, last_updated timestamps, etc.
        
        metadata = self.load_metadata()
        last_update = metadata.get("last_incremental_update")
        
        if last_update:
            last_update_dt = datetime.fromisoformat(last_update)
            hours_since_update = (datetime.now() - last_update_dt).total_seconds() / 3600
            
            if hours_since_update < 6:  # Don't update more than every 6 hours
                return {
                    "success": True,
                    "skipped": True,
                    "reason": f"Last update was {hours_since_update:.1f} hours ago (minimum 6 hours)"
                }
        
        # Run full refresh for incremental update
        result = self.run_full_refresh(create_backup=False)
        
        if result["success"]:
            metadata["last_incremental_update"] = datetime.now().isoformat()
            self.save_metadata(metadata)
        
        return result
    
    def setup_cron_job(self) -> str:
        """Generate a cron job command for automated updates."""
        script_path = Path(__file__).absolute()
        project_root = self.project_root.absolute()
        
        # Daily at 2 AM
        cron_command = f"0 2 * * * cd {project_root} && python {script_path} --mode incremental >> {project_root}/logs/auto_refresh.log 2>&1"
        
        cron_file = self.project_root / "cron_job.txt"
        with open(cron_file, 'w') as f:
            f.write(f"# Add this line to your crontab (crontab -e):\n")
            f.write(f"{cron_command}\n\n")
            f.write(f"# To install:\n")
            f.write(f"# crontab {cron_file}\n")
        
        print(f"📅 Cron job configuration saved to: {cron_file}")
        print(f"🔧 To install: crontab {cron_file}")
        
        return cron_command
    
    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        freshness = self.check_data_freshness()
        integrity = self.validate_data_integrity()
        metadata = self.load_metadata()
        
        report = f"""
📊 YELP DATA REFRESH STATUS REPORT
==================================

🕒 Data Freshness:
   Status: {freshness['status'].upper()}
   Message: {freshness['message']}
   Last Refresh: {freshness.get('last_refresh', 'Never')}
   Total Businesses: {int(freshness.get('total_businesses', 0)):,}
   Total Reviews: {int(freshness.get('total_reviews', 0)):,}

🔍 Data Integrity:
   Valid: {'✅ YES' if integrity['valid'] else '❌ NO'}
   Missing Files: {', '.join(integrity['missing_files']) or 'None'}
   Corrupted Files: {', '.join(integrity['corrupted_files']) or 'None'}

📁 File Statistics:
"""
        
        for file_name, stats in integrity['file_stats'].items():
            report += f"   {file_name}: {stats['rows']:,} rows, {stats['size_mb']:.1f} MB\n"
        
        if metadata.get('refresh_history'):
            report += f"\n📈 Recent Refresh History:\n"
            for record in metadata['refresh_history'][-3:]:  # Last 3
                report += f"   {record['timestamp'][:19]}: {record['type']} ({record['duration_minutes']:.1f} min)\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Automated Yelp Data Refresh System")
    parser.add_argument("--mode", choices=["full", "incremental", "check"], 
                       default="check", help="Refresh mode")
    parser.add_argument("--setup-cron", action="store_true", 
                       help="Generate cron job configuration")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip backup creation for full refresh")
    parser.add_argument("--project-root", type=Path, 
                       help="Project root directory")
    
    args = parser.parse_args()
    
    refresher = YelpDataRefresher(args.project_root)
    
    if args.setup_cron:
        refresher.setup_cron_job()
        return
    
    if args.mode == "check":
        print(refresher.generate_status_report())
        
    elif args.mode == "full":
        result = refresher.run_full_refresh(create_backup=not args.no_backup)
        if result["success"]:
            print(f"✅ Full refresh completed successfully!")
            print(f"   Businesses: {result['businesses_count']:,}")
            print(f"   Reviews: {result['reviews_count']:,}")
            print(f"   Duration: {result['duration_minutes']:.1f} minutes")
        else:
            print(f"❌ Full refresh failed: {result['error']}")
            sys.exit(1)
            
    elif args.mode == "incremental":
        result = refresher.run_incremental_update()
        if result["success"]:
            if result.get("skipped"):
                print(f"⏭️ {result['reason']}")
            else:
                print(f"✅ Incremental update completed!")
                print(f"   Businesses: {result['businesses_count']:,}")
                print(f"   Reviews: {result['reviews_count']:,}")
        else:
            print(f"❌ Incremental update failed: {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()
