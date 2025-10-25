#!/usr/bin/env python3
"""
Data Refresh Monitoring Dashboard
================================

This Streamlit page provides monitoring and control for automated Yelp data refresh.
Shows data freshness, refresh history, and allows manual triggers.

Features:
- Data freshness status
- Refresh history timeline
- Manual refresh triggers
- Data integrity checks
- Cron job status
- Performance metrics
"""

import streamlit as st
import pandas as pd
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from auto_refresh_data import YelpDataRefresher

def main():
    st.set_page_config(
        page_title="Data Refresh Monitor",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Data Refresh Monitoring Dashboard")
    st.markdown("Monitor and control automated Yelp data refresh")
    
    # Initialize refresher
    refresher = YelpDataRefresher()
    
    # Sidebar controls
    st.sidebar.header("🛠️ Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Full Refresh", type="primary"):
            with st.spinner("Running full refresh..."):
                result = refresher.run_full_refresh()
                if result["success"]:
                    st.success("✅ Full refresh completed!")
                    st.rerun()
                else:
                    st.error(f"❌ Refresh failed: {result['error']}")
    
    with col2:
        if st.button("⚡ Incremental Update"):
            with st.spinner("Running incremental update..."):
                result = refresher.run_incremental_update()
                if result["success"]:
                    if result.get("skipped"):
                        st.info(f"⏭️ {result['reason']}")
                    else:
                        st.success("✅ Incremental update completed!")
                    st.rerun()
                else:
                    st.error(f"❌ Update failed: {result['error']}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Status", "📈 History", "🔍 Integrity", "⚙️ Settings"])
    
    with tab1:
        st.header("📊 Data Status")
        
        # Data freshness
        freshness = refresher.check_data_freshness()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if freshness["status"] == "fresh":
                st.metric("Data Status", "🟢 Fresh", delta=f"{freshness['days_old']} days old")
            elif freshness["status"] == "stale":
                st.metric("Data Status", "🟡 Stale", delta=f"{freshness['days_old']} days old")
            else:
                st.metric("Data Status", "🔴 Outdated", delta=f"{freshness['days_old']} days old")
        
        with col2:
            st.metric("Total Businesses", int(freshness.get('total_businesses', 0)))
        
        with col3:
            st.metric("Total Reviews", int(freshness.get('total_reviews', 0)))
        
        # Status details
        st.subheader("Status Details")
        st.info(freshness["message"])
        
        if freshness.get("last_refresh"):
            last_refresh = datetime.fromisoformat(freshness["last_refresh"])
            st.write(f"**Last Refresh:** {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick actions based on status
        if freshness["status"] == "outdated":
            st.warning("⚠️ Data is outdated. Consider running a full refresh.")
        elif freshness["status"] == "stale":
            st.info("ℹ️ Data is getting stale. An incremental update might be helpful.")
    
    with tab2:
        st.header("📈 Refresh History")
        
        metadata = refresher.load_metadata()
        history = metadata.get("refresh_history", [])
        
        if history:
            # Convert to DataFrame for better display
            df_history = pd.DataFrame(history)
            df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
            df_history = df_history.sort_values("timestamp", ascending=False)
            
            # Display recent refreshes
            st.subheader("Recent Refreshes")
            for _, record in df_history.head(5).iterrows():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**{record['timestamp'].strftime('%Y-%m-%d %H:%M')}**")
                with col2:
                    st.write(f"Type: {record['type']}")
                with col3:
                    st.write(f"Duration: {record['duration_minutes']:.1f} min")
                with col4:
                    st.write(f"Businesses: {record['businesses_count']:,}")
            
            # Performance chart
            st.subheader("Performance Trends")
            chart_data = df_history[["timestamp", "duration_minutes", "businesses_count"]].copy()
            chart_data["timestamp"] = chart_data["timestamp"].dt.strftime("%m-%d %H:%M")
            
            st.line_chart(
                chart_data.set_index("timestamp")[["duration_minutes"]],
                height=300
            )
            
        else:
            st.info("No refresh history available. Run a refresh to see history.")
    
    with tab3:
        st.header("🔍 Data Integrity")
        
        integrity = refresher.validate_data_integrity()
        
        if integrity["valid"]:
            st.success("✅ All data files are valid and accessible")
        else:
            st.error("❌ Data integrity issues detected")
        
        # File statistics
        st.subheader("File Statistics")
        if integrity["file_stats"]:
            df_stats = pd.DataFrame(integrity["file_stats"]).T
            df_stats["size_mb"] = df_stats["size_mb"].round(2)
            st.dataframe(df_stats, use_container_width=True)
        
        # Issues
        if integrity["missing_files"]:
            st.error(f"**Missing Files:** {', '.join(integrity['missing_files'])}")
        
        if integrity["corrupted_files"]:
            st.error(f"**Corrupted Files:** {', '.join(integrity['corrupted_files'])}")
        
        # Manual integrity check
        if st.button("🔍 Re-check Integrity"):
            st.rerun()
    
    with tab4:
        st.header("⚙️ Settings & Automation")
        
        # Cron job status
        st.subheader("Cron Job Status")
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0 and "auto_refresh_data.py" in result.stdout:
                st.success("✅ Cron job is installed")
                
                # Show cron job details
                cron_lines = [line for line in result.stdout.split('\n') 
                             if "auto_refresh_data.py" in line]
                if cron_lines:
                    st.code(cron_lines[0], language="bash")
            else:
                st.warning("⚠️ No cron job installed")
                
        except Exception as e:
            st.error(f"❌ Error checking cron status: {e}")
        
        # Setup instructions
        st.subheader("Setup Instructions")
        st.markdown("""
        **To set up automated refresh:**
        
        1. **Daily refresh:**
           ```bash
           python src/setup_cron.py --schedule daily
           ```
        
        2. **Weekly refresh:**
           ```bash
           python src/setup_cron.py --schedule weekly
           ```
        
        3. **Custom schedule:**
           ```bash
           python src/setup_cron.py --schedule custom --custom-schedule "0 2 * * 1"
           ```
        
        4. **Remove cron job:**
           ```bash
           python src/setup_cron.py --remove
           ```
        """)
        
        # Manual commands
        st.subheader("Manual Commands")
        st.markdown("""
        **Command line options:**
        
        - Check status: `python src/auto_refresh_data.py --mode check`
        - Full refresh: `python src/auto_refresh_data.py --mode full`
        - Incremental update: `python src/auto_refresh_data.py --mode incremental`
        """)
        
        # Log monitoring
        st.subheader("Log Monitoring")
        log_file = Path("logs/auto_refresh.log")
        if log_file.exists():
            st.success(f"✅ Log file exists: {log_file}")
            
            # Show last few lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        st.subheader("Recent Log Entries")
                        st.code(''.join(lines[-10:]), language="text")
            except Exception as e:
                st.error(f"Error reading log: {e}")
        else:
            st.info("ℹ️ No log file found yet. Run a refresh to create logs.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

if __name__ == "__main__":
    main()
