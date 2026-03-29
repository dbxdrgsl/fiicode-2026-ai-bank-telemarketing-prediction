#!/bin/bash
# Check overnight experiment progress
# Usage: bash check_progress.sh

cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode
source .venv/bin/activate

python3 << 'EOF'
import sqlite3
from pathlib import Path
from datetime import datetime

print(f"\n{'='*60}")
print(f"EXPERIMENT PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

experiments = [
    ("exp033_target_encoding", 60),
    ("exp034_lightgbm_target_enc", 50),
    ("exp035_xgboost_target_enc", 50),
]

for exp_name, total_trials in experiments:
    db_path = Path(f"outputs/logs/{exp_name}/{exp_name}.sqlite3")
    
    if not db_path.exists():
        print(f"⏸️  {exp_name}: Not started")
        continue
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as count, 
               MAX(tv.value) as best 
        FROM trials t 
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
    """)
    count, best = cursor.fetchone()
    
    pct = (count / total_trials) * 100 if count else 0
    
    if count == total_trials:
        print(f"✅ {exp_name}: DONE ({count}/{total_trials}) - Best CV: {best:.5f}")
    elif count > 0:
        print(f"🏃 {exp_name}: Running ({count}/{total_trials}, {pct:.0f}%) - Best: {best:.5f}")
    else:
        print(f"⏳ {exp_name}: Starting...")
    
    conn.close()

print(f"\n{'='*60}\n")
EOF
