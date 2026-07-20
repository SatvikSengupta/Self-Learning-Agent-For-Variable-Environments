import subprocess
import sys
import time

def main():
    print("=======================================================")
    print("Starting RLHF Master Controller...")
    print("Press Ctrl+C in this terminal to shut down all systems.")
    print("=======================================================\n")

    train_process = None
    ui_process = None

    try:
        # 1. Launch the heavy training and environment process
        # Using sys.executable ensures it uses your current active virtual environment (.venv)
        train_process = subprocess.Popen([sys.executable, "train.py"])
        print("[System] Launched train.py (PyTorch & PyBullet)")

        # Give the physics engine and socket listener 2 seconds to initialize
        time.sleep(2)

        # 2. Launch the UI control panel
        ui_process = subprocess.Popen([sys.executable, "feedback_ui.py"])
        print("[System] Launched feedback_ui.py (Tkinter UI)")

        # Keep the master script alive and watching the child processes
        while train_process.poll() is None and ui_process.poll() is None:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[System] Termination signal received. Shutting down...")
    
    finally:
        # Gracefully kill both processes if they are still running
        if train_process and train_process.poll() is None:
            train_process.terminate()
            print("[System] Terminated train.py")
        
        if ui_process and ui_process.poll() is None:
            ui_process.terminate()
            print("[System] Terminated feedback_ui.py")
            
        print("[System] All systems offline.")

if __name__ == "__main__":
    main()