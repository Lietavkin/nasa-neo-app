"""
How to use:
- Run this script with: python schedule_fetch.py
- Leave it running in the background to automatically run daily_pipeline.py every day at 09:00 (server time).
- You can use tools like tmux, screen, or systemd to keep it running after logout if needed.
"""
import schedule
import subprocess
import time
from datetime import datetime


def run_daily_pipeline():
    print(f"[SCHEDULER] Job started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        subprocess.run(["python3", "daily_pipeline.py"], check=True)
        print(f"[SCHEDULER] Job finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except subprocess.CalledProcessError as e:
        print(f"[SCHEDULER] Job failed: {e}")

schedule.every().day.at("09:00").do(run_daily_pipeline)

print("[SCHEDULER] schedule_fetch.py started. Waiting for next scheduled run...")

while True:
    schedule.run_pending()
    time.sleep(30) 