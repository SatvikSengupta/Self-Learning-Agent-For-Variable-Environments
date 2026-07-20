import tkinter as tk
import socket

# Configure the UDP Socket
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_reward(val):
    # Convert the integer to bytes and broadcast it locally
    sock.sendto(str(val).encode('utf-8'), (UDP_IP, UDP_PORT))
    print(f"Signal Sent: {val}")

# Build the GUI
root = tk.Tk()
root.title("RLHF Control Panel")
root.geometry("300x400")
root.configure(bg="#2b2b2b")

# Instructions
label = tk.Label(root, text="Agent Feedback", fg="white", bg="#2b2b2b", font=("Arial", 16, "bold"))
label.pack(pady=20)

# The Three Buttons
btn_good = tk.Button(root, text="GOOD (+1)", bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), 
                     height=3, command=lambda: send_reward(1.0))
btn_good.pack(fill=tk.X, padx=20, pady=10)

btn_okay = tk.Button(root, text="OKAY (0)", bg="#9E9E9E", fg="white", font=("Arial", 14, "bold"), 
                     height=3, command=lambda: send_reward(0.0))
btn_okay.pack(fill=tk.X, padx=20, pady=10)

btn_no = tk.Button(root, text="NO (-1)", bg="#F44336", fg="white", font=("Arial", 14, "bold"), 
                   height=3, command=lambda: send_reward(-1.0))
btn_no.pack(fill=tk.X, padx=20, pady=10)

print("Control Panel Active. Ready to send signals.")
root.mainloop()