import subprocess
import sys

steps = [
    ("EDA", "src/eda.py"),
    ("Data Preprocessing", "src/preprocessing.py"),
    ("Feature Engineering", "src/feature_engineering.py"),
    ("Model Training", "src/train_model.py"),
    ("Model Evaluation", "src/evaluate_model.py"),
]

for step_name, script in steps:
    print("\n" + "=" * 70)
    print(f"Running {step_name}")
    print("=" * 70)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\nError while running {script}")
        break

print("\nProject Execution Finished!")