#!/usr/bin/env python3
"""
Cron Job Setup for Automated Yelp Data Refresh
==============================================

This script helps you set up automated data refresh using cron jobs.
It provides different scheduling options and handles logging.

Usage:
    python src/setup_cron.py --schedule daily
    python src/setup_cron.py --schedule weekly
    python src/setup_cron.py --schedule custom "0 2 * * 1"  # Every Monday at 2 AM
    python src/setup_cron.py --remove  # Remove existing cron job
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def get_cron_job_command(schedule: str, custom_schedule: str = None) -> str:
    """Generate cron job command based on schedule."""
    project_root = Path(__file__).parent.parent.absolute()
    script_path = project_root / "src" / "auto_refresh_data.py"
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Schedule mappings
    schedules = {
        "daily": "0 2 * * *",      # Daily at 2 AM
        "weekly": "0 2 * * 1",      # Weekly on Monday at 2 AM
        "twice_daily": "0 2,14 * * *",  # Twice daily at 2 AM and 2 PM
        "hourly": "0 * * * *",      # Every hour
        "custom": custom_schedule
    }
    
    cron_schedule = schedules.get(schedule)
    if not cron_schedule:
        raise ValueError(f"Invalid schedule: {schedule}")
    
    # Build the command
    command = f"cd {project_root} && python {script_path} --mode incremental"
    log_file = log_dir / "auto_refresh.log"
    
    cron_job = f"{cron_schedule} {command} >> {log_file} 2>&1"
    
    return cron_job

def install_cron_job(cron_command: str) -> bool:
    """Install the cron job."""
    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        # Check if our job already exists
        if "auto_refresh_data.py" in current_crontab:
            print("⚠️  Cron job already exists. Removing old one first...")
            remove_cron_job()
        
        # Add new job
        new_crontab = current_crontab.rstrip() + "\n" + cron_command + "\n"
        
        # Install new crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(input=new_crontab)
        
        if process.returncode == 0:
            print("✅ Cron job installed successfully!")
            return True
        else:
            print("❌ Failed to install cron job")
            return False
            
    except Exception as e:
        print(f"❌ Error installing cron job: {e}")
        return False

def remove_cron_job() -> bool:
    """Remove the existing cron job."""
    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ℹ️  No existing crontab found")
            return True
        
        current_crontab = result.stdout
        lines = current_crontab.split('\n')
        
        # Filter out our job
        filtered_lines = [line for line in lines if "auto_refresh_data.py" not in line]
        
        if len(filtered_lines) == len(lines):
            print("ℹ️  No existing cron job found to remove")
            return True
        
        # Install filtered crontab
        new_crontab = '\n'.join(filtered_lines)
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(input=new_crontab)
        
        if process.returncode == 0:
            print("✅ Cron job removed successfully!")
            return True
        else:
            print("❌ Failed to remove cron job")
            return False
            
    except Exception as e:
        print(f"❌ Error removing cron job: {e}")
        return False

def list_cron_jobs():
    """List current cron jobs."""
    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📅 Current cron jobs:")
            print(result.stdout)
        else:
            print("ℹ️  No cron jobs found")
    except Exception as e:
        print(f"❌ Error listing cron jobs: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup automated Yelp data refresh")
    parser.add_argument("--schedule", choices=["daily", "weekly", "twice_daily", "hourly", "custom"],
                       help="Schedule frequency")
    parser.add_argument("--custom-schedule", type=str,
                       help="Custom cron schedule (e.g., '0 2 * * 1' for Monday 2 AM)")
    parser.add_argument("--remove", action="store_true",
                       help="Remove existing cron job")
    parser.add_argument("--list", action="store_true",
                       help="List current cron jobs")
    
    args = parser.parse_args()
    
    if args.list:
        list_cron_jobs()
        return
    
    if args.remove:
        remove_cron_job()
        return
    
    if not args.schedule:
        print("❌ Please specify --schedule or use --remove/--list")
        sys.exit(1)
    
    try:
        cron_command = get_cron_job_command(args.schedule, args.custom_schedule)
        
        print(f"🔧 Setting up {args.schedule} data refresh...")
        print(f"📅 Cron command: {cron_command}")
        
        if install_cron_job(cron_command):
            print("\n📋 Next steps:")
            print("1. Check logs: tail -f logs/auto_refresh.log")
            print("2. Test manually: python src/auto_refresh_data.py --mode incremental")
            print("3. Check status: python src/auto_refresh_data.py --mode check")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
