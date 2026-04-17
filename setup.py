"""
VentureX — Full Setup & Build Script
Run this ONCE to set everything up:
    python setup.py
Then launch the app:
    streamlit run app.py
"""

import os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def run(cmd_list, cwd=BASE):
    print(f"\n$ {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=cwd)
    if result.returncode != 0:
        print(f"⚠ Command exited with code {result.returncode}")
    return result.returncode

def main():
    print("=" * 55)
    print("  VentureX — Setup")
    print("  Rishi Ponda · Anmol Singh · Priyanshu Padhi")
    print("=" * 55)

    py = sys.executable  # uses the exact running interpreter (no shell quoting issues)

    # 1. Install dependencies
    print("\n[1/3] Installing dependencies …")
    run([py, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

    # 2. Generate dataset
    print("\n[2/3] Generating startup dataset …")
    run([py, "data/generate_data.py"])

    # 3. Train ML models
    print("\n[3/3] Training ML models …")
    run([py, "models/train.py"])

    print("\n" + "=" * 55)
    print("✅ Setup complete!")
    print("\n▶ Run the app:")
    print("    streamlit run app.py")
    print("=" * 55)

if __name__ == "__main__":
    main()
