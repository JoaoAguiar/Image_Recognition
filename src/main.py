import sys
import subprocess
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <algorithm>")
        print("Algorithms: rf, knn, dt, bg, ab")
        return

    algorithm = sys.argv[1]
    script_map = {
        "rf": "rf_recognition.py",
        "knn": "knn_recognition.py",
        "dt": "dt_recognition.py",
        "bg": "bg_recognition.py",
        "ab": "ab_recognition.py"
    }

    if algorithm in script_map:
        script_name = script_map[algorithm]
        # Ensure the output directory exists
        os.makedirs("../output", exist_ok=True)
        subprocess.run(["python", script_name])
    else:
        print("Invalid algorithm. Choose from: rf, knn, dt, bg, ab")

if __name__ == "__main__":
    main()
