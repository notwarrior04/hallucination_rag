import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_benchmark_pid():
    """Gets the PID of the running benchmark_ragtruth.py process."""
    try:
        out = subprocess.check_output("wmic process where \"name='python.exe'\" get commandline,processid", shell=True, text=True)
        for line in out.splitlines():
            if "benchmark_ragtruth.py" in line and "run_full_evaluation.py" not in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit():
                        return int(pid)
    except Exception as e:
        logger.warning(f"Error querying process via wmic: {e}")
    return None

def is_pid_running(pid):
    """Checks if a process with the given PID is still running."""
    try:
        out = subprocess.check_output(f"tasklist /fi \"PID eq {pid}\"", shell=True, text=True)
        # tasklist outputs a table containing the PID if the process exists
        return str(pid) in out
    except Exception:
        return False

def run_command(args, log_file_path=None):
    """Runs a shell command and logs its output in real-time."""
    logger.info(f"Running command: {' '.join(args)}")
    
    # Ensure parent directories exist
    if log_file_path:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_file_path, "w", encoding="utf-8")
    else:
        log_file = sys.stdout

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_file_path:
            log_file.write(line)
            log_file.flush()

    process.wait()
    if log_file_path:
        log_file.close()
    
    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    logger.info("Command completed successfully.")

def main():
    python_path = sys.executable
    logger.info(f"Using Python executable: {python_path}")

    # Step 1: Wait for RAGTruth benchmark to complete
    logger.info("Checking if RAGTruth benchmark (benchmark_ragtruth.py) is running...")
    while True:
        pid = get_benchmark_pid()
        if pid and is_pid_running(pid):
            logger.info(f"RAGTruth benchmark is running with PID {pid}. Waiting 30 seconds...")
            time.sleep(30)
        else:
            logger.info("RAGTruth benchmark is not running. Proceeding.")
            break

    # Verify RAGTruth outputs are present
    ragtruth_metrics_path = Path("evaluation/results/ragtruth_metrics.json")
    if not ragtruth_metrics_path.exists():
        logger.warning("RAGTruth metrics file does not exist. Running RAGTruth benchmark now...")
        run_command([
            python_path,
            "evaluation/benchmark_ragtruth.py",
            "--max_samples", "900"
        ], "evaluation/results/ragtruth_run.log")
    else:
        logger.info("RAGTruth benchmark metrics found.")

    # Step 2: Run HaluEval Benchmark (10,000 samples)
    # HaluEval loading loads 3 subsets, args.max_samples per subset.
    # Each raw sample yields 1 factual and 1 hallucinated sample, contributing 2 items to the balanced evaluation suite.
    # To get 10,000 samples in total, we set max_samples to 1667 (1667 * 3 * 2 = 10,002 evaluation samples)
    logger.info("Starting Full HaluEval benchmark (10,000 evaluation samples)...")
    run_command([
        python_path,
        "evaluation/benchmark_halueval.py",
        "--max_samples", "1667"
    ], "evaluation/results/halueval_run.log")

    # Step 3: Run Calibration Evaluation (1,000 samples)
    logger.info("Starting Calibration evaluation (1,000 samples)...")
    run_command([
        python_path,
        "evaluation/calibration_eval.py",
        "--max_samples", "1000"
    ], "evaluation/results/calibration_run.log")

    # Step 4: Run Length-Based Bucket Analysis
    logger.info("Starting Length Bucket Analysis...")
    run_command([
        python_path,
        "evaluation/length_analysis.py"
    ], "evaluation/results/length_analysis_run.log")

    logger.info("=" * 60)
    logger.info("ALL EVALUATIONS AND ANALYSES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
