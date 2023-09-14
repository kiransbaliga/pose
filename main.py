import subprocess

# Define the file paths for the three Python files you want to run sequentially
file1_path = "fresh.py"
file2_path = "fresh_squat.py"
file3_path = "fresh_pushup.py"

# Define a function to run a Python file using subprocess
def run_python_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
        print(f"Successfully ran {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file_path}: {e}")

# Run the three Python files sequentially
run_python_file(file1_path)
run_python_file(file2_path)
run_python_file(file3_path)
