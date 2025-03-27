import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import serial
from threading import Thread

root = tk.Tk()
valori = []

try:
    ser = serial.Serial('COM12', 115200)  # open serial port
    ser.readline()
except:
    print("Serial port inexistent")

def serialReading():
    global valori, ser
    while(running):
        linie = ser.readline()
        valori = list(linie)

class MicSignal:
    def __init__(self, root):
        self.root = root
        self.root.title("Microphone SIE")

        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        self.data = None
        self.fft_data = None
        self.sampling_rate = 25000  
        
        self.left_frame = tk.Frame(root, width=200, bg="#e0e0e0", padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        


        self.plot_frame1 = tk.Frame(self.right_frame, bg="white", highlightbackground="gray", highlightthickness=1)
        self.plot_frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.plot_frame2 = tk.Frame(self.right_frame, bg="white", highlightbackground="gray", highlightthickness=1)
        self.plot_frame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")


        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        


        self.load_button = tk.Button(self.left_frame, text="Load", command=self.load_data, 
                                     bg="#4CAF50", fg="white", font=("Arial", 12),
                                     padx=20, pady=10, width=15)
        self.load_button.pack(pady=(20, 10))
        
        self.save_button = tk.Button(self.left_frame, text="Save", command=self.save_fft, 
                                     bg="#2196F3", fg="white", font=("Arial", 12),
                                     padx=20, pady=10, width=15)
        self.save_button.pack(pady=10)
        

        self.create_empty_plots()
        
    def create_empty_plots(self):
        self.fig1 = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Original Signal")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True)
        
        # Adăugăm spațiu pentru etichete
        self.fig1.subplots_adjust(bottom=0.15)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.plot_frame1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("FFT")
        self.ax2.set_xlabel("Frequency")
        self.ax2.set_ylabel("Magnitude")
        self.ax2.grid(True)
        
        # Adăugăm spațiu pentru etichete
        self.fig2.subplots_adjust(bottom=0.15)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.plot_frame2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as file:
                data = []
                for line in file:
                    try:
                        line = line.strip()
                        if ',' in line:
                            values = [float(x.strip()) for x in line.split(',')]
                        else:
                            values = [float(x) for x in line.split()]
                        data.extend(values)
                    except ValueError:
                        continue
                print(data)
            
            # Primele 1000 valori
            self.data = np.array(data[:1000])
            if len(self.data) == 0:
                messagebox.showerror("Error", "No data found.")
                return
                
            if len(self.data) < 1000:
                messagebox.showwarning("Warning", f"Only {len(self.data)} values were found in the file.")
            
            #print(self.data)
            self.plot1(self.data)
            self.calculate_fft()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def plot1(self, data_array):
        self.ax1.clear()
        self.ax1.plot(data_array)
        self.ax1.set_title("Original Signal")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True)
        self.canvas1.draw()
    
    def calculate_fft(self):
        if self.data is None or len(self.data) == 0:
            return

        self.fft_data = np.abs(np.fft.fft(self.data))
        
        freqs = np.fft.fftfreq(len(self.data), 1/self.sampling_rate)
        
        half_len = len(freqs) // 2
        self.fft_data = self.fft_data[75:half_len]
        freqs = freqs[75:half_len]
        
        self.ax2.clear()
        
        max_freq = max(freqs)
        if max_freq > 1e6:
            plot_freqs = freqs / 1e6
            freq_unit = "MHz"
        elif max_freq > 1e3:
            plot_freqs = freqs / 1e3
            freq_unit = "kHz"
        else:
            plot_freqs = freqs
            freq_unit = "Hz"
        
        self.ax2.plot(plot_freqs, self.fft_data)
        self.ax2.set_title("FFT")
        
        self.ax2.set_xlabel(f"Frequency ({freq_unit})", fontsize=12)
        self.ax2.set_ylabel("Magnitude")
        self.ax2.grid(True)
        
        self.fig2.subplots_adjust(bottom=0.15)
        self.canvas2.draw()
    
    def save_fft(self):
        if self.fft_data is None:
            messagebox.showwarning("Warning", "No FFT data to save. Please load data first.")
            return
        
        # Open file dialog to select save location
        file_path = filedialog.asksaveasfilename(
            title="Save FFT Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.fig2.savefig(file_path)
            messagebox.showinfo("Success", f"FFT plot saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")




running = True

app = MicSignal(root)


def updateGraph():
    #print("ALO")
    global app
    app.data = valori
    app.plot1(np.array(valori[:1000]))
    app.calculate_fft()
    root.after(10, updateGraph)

if __name__ == "__main__":
    t = Thread(target=serialReading)
    t.start()
    root.after(10, updateGraph)
    root.mainloop()
    running = False
    t.join()
