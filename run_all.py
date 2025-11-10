import subprocess
from pathlib import Path
import sys

def main():
    
    input_file = sys.argv[1]
    base_name = Path(input_file).stem
    test_mode = "--test" in sys.argv
    test_flag = ["--test"] if test_mode else []

    # Defines the pipeline of which files to run in what order
    pipeline = [
        ("scrape_jobs.py", [f"{base_name}.txt"]),
        ("rate_job_fit.py", [f"{base_name}"]),
        ("application_workflow.py", [f"{base_name}"])
    ]

    for script, script_args in pipeline:
        cmd = ["python", script] + script_args + test_flag
        print(f"\n🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"⚠️ {script} failed (exit code {result.returncode}). Stopping pipeline.")
            break
        print(f"✅ Finished {script}\n")

if __name__ == "__main__":
    main()
