import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import serial
import threading
import time
import collections
import struct
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks  


class MicSignal:
    def __init__(self, root):
        self.root = root
        self.root.title("Microphone SIE")

        self.root.geometry("1200x1000")
        self.root.configure(bg="white")
        
        self.data = None
        self.fft_data = None
        self.fft_freqs = None  
        self.sampling_rate = 25000  
        self.serial_port = None
        self.recording = False
        self.default_com_port = "COM7"  #setare com pentru date
        
        self.vowels = ['a', 'e', 'i', 'o', 'u']
        self.vowel_models = {}
        
        # Cheia de criptare pentru XTEA (16 bytes)
        self.encryption_key = b"0123456789abcdef"

        self.load_vowel_models()
        
        # buffer circular pentru datele de la microfon
        self.sample_buffer = collections.deque(maxlen=1000)
        # thread separat pentru a putea citi date constat de la microfon cat timp 
        self.reading_thread = None
        self.stop_reading = False
        
        self.left_frame = tk.Frame(root, width=200, bg="gray95", padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(root, bg="gray95", padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.plot_frame1 = tk.Frame(self.right_frame, bg="white", highlightbackground="gray", highlightthickness=1)
        self.plot_frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.plot_frame2 = tk.Frame(self.right_frame, bg="white", highlightbackground="gray", highlightthickness=1)
        self.plot_frame2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # frame pentru rezultate
        self.recognition_frame = tk.LabelFrame(self.right_frame, text="Recunoastere vocala", 
                                      bg="white", font=("Arial", 12, "bold"), padx=10, pady=10)
        self.recognition_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # label voacale
        self.vowel_label = tk.Label(self.recognition_frame, text="Nicio vocala detectata", 
                                    font=("Arial", 20, "bold"), bg="white")
        self.vowel_label.pack(pady=10)
        
        # frame bare progres
        self.progress_frame = tk.Frame(self.recognition_frame, bg="white")
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bars = {}
        self.progress_labels = {}
        
        for i, vowel in enumerate(self.vowels):
            frame = tk.Frame(self.progress_frame, bg="white")
            frame.pack(fill=tk.X, pady=5)
            
            label = tk.Label(frame, text=f"Vocala: {vowel.upper()}", width=10, font=("Arial", 9), bg="white")
            label.pack(side=tk.LEFT)
            
            progress = tk.Canvas(frame, width=300, height=20, bg="#f0f0f0", highlightthickness=1)
            progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            value_label = tk.Label(frame, text="0%", width=8, font=("Arial", 10), bg="white")
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.progress_bars[vowel] = progress
            self.progress_labels[vowel] = value_label

        self.right_frame.grid_rowconfigure(0, weight=2)
        self.right_frame.grid_rowconfigure(1, weight=2)
        self.right_frame.grid_rowconfigure(2, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # selector com port
        self.port_frame = tk.Frame(self.left_frame, bg="gray95")
        self.port_frame.pack(pady=(0, 10), fill=tk.X)
        
        port_label = tk.Label(self.port_frame, text="Port COM:", bg="gray95", font=("Arial", 10))
        port_label.pack(anchor=tk.W)
        
        # input com
        self.port_var = tk.StringVar(value=self.default_com_port)
        self.port_entry = tk.Entry(self.port_frame, textvariable=self.port_var, font=("Arial", 10))
        self.port_entry.pack(fill=tk.X, pady=5)
        
        self.connect_button = tk.Button(self.port_frame, text="Connect", command=self.connect_serial,
                                     bg="firebrick1", fg="white", font=("Arial", 10))
        self.connect_button.pack(fill=tk.X)

        # butoane save/load
        self.load_button = tk.Button(self.left_frame, text="Load", command=self.load_data, 
                                     bg="green3", fg="white", font=("Arial", 12),
                                     padx=20, pady=10, width=15)
        self.load_button.pack(pady=(20, 10))
        
        self.save_button = tk.Button(self.left_frame, text="Save", command=self.save_fft, 
                                     bg="royalblue1", fg="white", font=("Arial", 12),
                                     padx=20, pady=10, width=15)
        self.save_button.pack(pady=10)
        
        # buton de record
        self.record_button = tk.Button(self.left_frame, text="Record", command=self.record_audio, 
                                     bg="firebrick2", fg="white", font=("Arial", 12),
                                     padx=20, pady=10, width=15)
        self.record_button.pack(pady=10)
        self.record_button["state"] = "disabled"
        
        # buton pentru calcularea mean-vectorilor
        self.calculate_means_button = tk.Button(self.left_frame, text="Calculate Mean Vectors", command=self.calculate_mean_vectors,
                                      bg="darkorchid2", fg="white", font=("Arial", 12),
                                      padx=20, pady=10, width=15)
        self.calculate_means_button.pack(pady=10)
        
        self.create_empty_plots()
                
    def connect_serial(self):
        if self.serial_port and self.serial_port.is_open:
            # deconectare
            try:
                # stop fir citire
                if self.reading_thread and self.reading_thread.is_alive():
                    self.stop_reading = True
                    self.reading_thread.join(timeout=1.0)
                
                self.serial_port.close()
                self.serial_port = None
                self.connect_button.config(text="Connect")
                self.record_button["state"] = "disabled"
                self.sample_buffer.clear()
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la deconectare: {str(e)}")
        else:
            # conectare
            port = self.port_var.get()
            if not port:
                messagebox.showwarning("Eroare", "Port-ul nu este valid")
                return
                
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                time.sleep(0.5)  #timp pt conexiune vazusem cv online cica e necesar
                
                self.connect_button.config(text="Disconnect")
                self.record_button["state"] = "disabled"
                self.sample_buffer.clear()
                
                # pornim thread-ul pentru citire
                self.stop_reading = False
                self.reading_thread = threading.Thread(target=self.continuous_read_thread, daemon=True)
                self.reading_thread.start()
                    
            except Exception as e:
                messagebox.showerror("Eroare", f"Eroare la conectare la {port}: {str(e)}")
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()
                    self.serial_port = None
    
    def continuous_read_thread(self):
        # thread citire de la pico constant
        while not self.stop_reading and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                
                try:
                    
                    value = int(line)
                    self.sample_buffer.append(value)
                    
                    # activare buton cand sunt date
                    if len(self.sample_buffer) > 100 and self.record_button["state"] == "disabled":
                        self.root.after(0, lambda: self.set_record_button_state("normal"))
                except ValueError:
                    pass
                        
            except Exception as e:
                pass
                
    
    def set_record_button_state(self, state):
        self.record_button["state"] = state
    

    
    def record_audio(self):
        if not self.serial_port or not self.serial_port.is_open:
            messagebox.showwarning("Avertisment", "Nu sunteti conecatat la pico")
            return
            
        if self.recording:
            return
            
        if len(self.sample_buffer) < 100: 
            messagebox.showwarning("Avertisment", "Nu sunt destule esantioane.")
            return
            
        self.recording = True
        self.record_button["state"] = "disabled"
        
        samples = list(self.sample_buffer)
        
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.txt")
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(f"{sample}\n")
                
        #actualizare grafice
        self.data = np.array(samples)
        self.update_plots()
        
        # identifica vocala
        recognized_vowel = self.identify_vowel()
        
 
        # reset starea
        self.recording = False
        self.record_button["state"] = "normal"

    def update_plots(self):
        if self.data is None or len(self.data) == 0:
            return

        self.ax1.clear()
        self.ax1.plot(self.data)
        self.ax1.set_title("Semnal original")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("Amplitudine")
        self.ax1.grid(True)
        self.canvas1.draw()
        
        self.calculate_fft()
        
    def create_empty_plots(self):
        self.fig1 = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Semanl Original")
        self.ax1.set_xlabel("Sample")
        self.ax1.set_ylabel("Amplitudine")
        self.ax1.grid(True)
        
        # spatiu etichete
        self.fig1.subplots_adjust(bottom=0.15)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.plot_frame1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("FFT")
        self.ax2.set_xlabel("Frecventa")
        self.ax2.set_ylabel("Magnitudine")
        self.ax2.grid(True)
        
        # etichete
        self.fig2.subplots_adjust(bottom=0.15)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.plot_frame2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# -------------------------------------------------------------
        # adaugare informatii pentru hover
        self.freq_info_text = self.fig2.text(0.02, 0.02, "", fontsize=9, 
                                             bbox=dict(facecolor='white', alpha=0.7))
        
        # conectare evenimente pentru hover
        self.canvas2.mpl_connect('motion_notify_event', self.on_hover_fft)
        
    def on_hover_fft(self, event):
        if event.inaxes == self.ax2:
            if self.fft_data is not None and self.fft_freqs is not None:
                # obtine valorile x si y de la pozitia mouse-ului
                x, y = event.xdata, event.ydata
                
                # calculeaza unitatea de frecventa si ajusteaza valorile
                max_freq = max(self.fft_freqs)
                if max_freq > 1e6:
                    unit = "MHz"
                    x_scaled = x * 1e6
                elif max_freq > 1e3:
                    unit = "kHz"
                    x_scaled = x * 1e3
                else:
                    unit = "Hz"
                    x_scaled = x
                
                # gaseste cel mai apropiat punct de date
                if self.fft_freqs.size > 0:
                    # convertim x_scaled inapoi la frecventa reala pentru a gasi indexul corect
                    x_idx = np.argmin(np.abs(self.fft_freqs - x_scaled))
                    freq = self.fft_freqs[x_idx]
                    amp = self.fft_data[x_idx]
                    
                    if max_freq > 1e6:
                        freq_display = f"{freq/1e6:.2f} {unit}"
                    elif max_freq > 1e3:
                        freq_display = f"{freq/1e3:.2f} {unit}"
                    else:
                        freq_display = f"{freq:.2f} {unit}"
                        
                    # actualizare text
                    self.freq_info_text.set_text(f"Frecventa: {freq_display}\nAmplitudine: {amp:.4f}")
                    
                    self.fig2.canvas.draw_idle()
            else:
                self.freq_info_text.set_text("")
                self.fig2.canvas.draw_idle()
        else:
            self.freq_info_text.set_text("")
            self.fig2.canvas.draw_idle()
# -------------------------------------------------------------

    
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
            

            self.data = np.array(data[:1000])
            if len(self.data) == 0:
                messagebox.showerror("Error", "No data found.")
                return
                
            if len(self.data) < 1000:
                messagebox.showwarning("Warning", f"Doar {len(self.data)} valori au fost gasite.")
            
            self.ax1.clear()
            self.ax1.plot(self.data)
            self.ax1.set_title("Semnal Original")
            self.ax1.set_xlabel("Sample")
            self.ax1.set_ylabel("Amplitudine")
            self.ax1.grid(True)
            self.canvas1.draw()
            
            self.calculate_fft()
            
        except Exception as e:
            messagebox.showerror("Error", f"Eroare incarcare fisier: {str(e)}")
    
    def calculate_fft(self):
        if self.data is None:
            return

        # normalizare semnal
        data = self.data - np.mean(self.data)
        if np.std(data) > 0:
            data = data / np.std(data)
            
        # hamming inainte de FFT ceva scurgere spectrala parca
        window = np.hamming(len(data))
        windowed_data = data * window
        
        # FFT pe datele cu fereastra aplicata
        self.fft_data = np.abs(np.fft.fft(windowed_data))
        
        freqs = np.fft.fftfreq(len(self.data), 1/self.sampling_rate)
        
        half_len = len(freqs) // 2
        self.fft_data = self.fft_data[:half_len]
        # --
        # self.fft_data = self.fft_data[75:half_len]
        # --

        # normalizare vector fft
        norm = np.linalg.norm(self.fft_data)
        if norm > 0:
            self.fft_data = self.fft_data / norm

        # --
        #  freqs = freqs[75:half_len]
        # --
        freqs = freqs[:half_len]

        self.fft_freqs = freqs
        
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
        
        self.ax2.set_xlabel(f"Frecventa ({freq_unit})", fontsize=12)
        self.ax2.set_ylabel("Magnitudine")
        self.ax2.grid(True)
        
        # height pentru a filtra peak-urile slabe si distance pentru a evita peak-uri prea apropiate
        
        #  prag minim de detectie (30/100 din valoarea )
        height_threshold = 0.3 * np.max(self.fft_data)
        
        # Distanta minima intre peak-uri (in puncte)
        min_distance = len(self.fft_data) // 50
        
        peak_indices, _ = find_peaks(self.fft_data, height=height_threshold, distance=min_distance)
        
        max_peaks_to_show = 5
        if len(peak_indices) > max_peaks_to_show:
            sorted_peaks = sorted(peak_indices, key=lambda idx: self.fft_data[idx], reverse=True)
            peak_indices = sorted_peaks[:max_peaks_to_show]
        
        for idx in peak_indices:
            freq_value = freqs[idx]
            mag_value = self.fft_data[idx]
            
            if max_freq > 1e6:
                freq_display = f"{freq_value/1e6:.1f}"
            elif max_freq > 1e3:
                freq_display = f"{freq_value/1e3:.1f}"
            else:
                freq_display = f"{freq_value:.1f}"
                
            self.ax2.plot(plot_freqs[idx], mag_value, "x", color='red', markersize=5)
            self.ax2.annotate(
                f"{freq_display}\n{mag_value:.3f}", 
                (plot_freqs[idx], mag_value),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white")
            )
        
        self.fig2.subplots_adjust(bottom=0.15)
        self.canvas2.draw()
    
    def save_fft(self):
        if self.fft_data is None:
            messagebox.showwarning("Warning", "Nu exista FFT de salvat. Incarca datele intai.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save FFT Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        
        if not file_path:
            return
        
        try:
            self.fig2.savefig(file_path)
            messagebox.showinfo("Succes", f"Grafic fft salvat {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Eroare: {str(e)}")
    
    def load_vowel_models(self):
        vowel_dirs = {
            'a': os.path.join(os.path.dirname(os.path.abspath(__file__)), "a", "mean_vector_a.txt"),
            'e': os.path.join(os.path.dirname(os.path.abspath(__file__)), "e", "mean_vector_e.txt"),
            'i': os.path.join(os.path.dirname(os.path.abspath(__file__)), "i", "mean_vector_i.txt"),
            'o': os.path.join(os.path.dirname(os.path.abspath(__file__)), "o", "mean_vector_o.txt"),
            'u': os.path.join(os.path.dirname(os.path.abspath(__file__)), "u", "mean_vector_u.txt")
        }
        
        for vowel, path in vowel_dirs.items():
            try:
                if os.path.exists(path):
                    print(f"Loading vowel model from: {path}")
                    # datele din mean vect
                    with open(path, 'r') as f:
                        values = []
                        for line in f:
                            try:
                                value = float(line.strip())
                                values.append(value)
                            except ValueError:
                                pass
                    
                    if values:
                        # normalizare semnal - imbunatatit
                        data = np.array(values[:1000])
                        # Eliminare DC offset
                        data = data - np.mean(data)
                        # Normalizare amplitude
                        if np.std(data) > 0:
                            data = data / np.std(data)
                        
                        #  FFT  doar prima jumatate
                        fft_data = np.abs(np.fft.fft(data))
                        half_len = len(fft_data) // 2
                        
                        # --
                        # fft_data = fft_data[75:half_len]
                        # --


                        fft_data = fft_data[:half_len]
                        
                        window_size = len(fft_data)
                        window = np.hamming(window_size)
                        fft_data = fft_data * window
                        
                        norm = np.linalg.norm(fft_data)
                        if norm > 0:
                            fft_data = fft_data / norm
                            
                        self.vowel_models[vowel] = fft_data

            except Exception as e:
                print(f"Eroare incarcare model {vowel}: {str(e)}")
    
    def calculate_mean_vectors(self):
       
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            for vowel in self.vowels:
                vowel_dir = os.path.join(base_dir, vowel)
                
                if not os.path.exists(vowel_dir):
                    messagebox.showwarning("Avertisment", f"Folderul pentru vocala '{vowel}' nu exista.")
                    continue
                
                # exclude mean vect
                sample_files = [f for f in os.listdir(vowel_dir) 
                              if f.endswith('.txt') and not f.startswith('mean_vector_')]
                
                if not sample_files:
                    messagebox.showwarning("Avertisment", f"Nu exista fisiere de esantioane pentru vocala '{vowel}'.")
                    continue
                
            
                all_vectors = []
                
                for sample_file in sample_files:
                    file_path = os.path.join(vowel_dir, sample_file)
                    try:
                        with open(file_path, 'r') as f:
                            data = []
                            for line in f:
                                try:
                                    value = float(line.strip())
                                    data.append(value)
                                except ValueError:
                                    pass
                        
                        if not data:
                            continue
                        
                        all_vectors.append(data)
                    
                    except Exception as e:
                        print(f"Eroare la procesarea fisierului {sample_file}: {str(e)}")
                        continue
                
                if not all_vectors:
                    messagebox.showwarning("Avertisment", 
                                          f"Nu s-au putut procesa fisierele pentru vocala '{vowel}'.")
                    continue
                
                min_length = min(len(v) for v in all_vectors)
                all_vectors = [v[:min_length] for v in all_vectors]
                
                mean_vector = np.mean(all_vectors, axis=0)
                
                output_path = os.path.join(vowel_dir, f"mean_vector_{vowel}.txt")
                with open(output_path, 'w') as f:
                    for value in mean_vector:
                        f.write(f"{value}\n")
                
                print(f"Vector mediu pentru vocala '{vowel}' salvat in: {output_path}")
            
            self.load_vowel_models()
            messagebox.showinfo("Succes", "Vectorii medii au fost calculati si salvati pentru toate vocalele.")
            
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la calcularea vectorilor medii: {str(e)}")
    
    def update_confidence_display(self, distances):
        if not distances:
            return
            
        min_vowel = min(distances.items(), key=lambda x: x[1])[0]
        
        #  0 = 100% incredere
        confidence = {vowel: max(0, min(100, int((1 - dist) * 100))) 
                     for vowel, dist in distances.items()}
        
        self.vowel_label.config(text=f"Vocala detectata: {min_vowel.upper()}")
        
        # actualizare bare progress
        for vowel, conf in confidence.items():
            progress = self.progress_bars[vowel]
            progress.delete("all")

            width = progress.winfo_width() - 2
            if width < 1:
                width = 300  
            fill_width = int(width * conf / 100)
            
            if vowel == min_vowel:
                # Vocala cea mai aproape de aia detectata
                color = "#4CAF50"  
            else:
                # celelattle
                color = "#FF9800"  
            
            progress.create_rectangle(1, 1, fill_width, 19, fill=color, outline="")
            progress.create_rectangle(1, 1, width, 19, outline="#CCCCCC")
            
            self.progress_labels[vowel].config(text=f"{conf}%")
    

    def identify_vowel(self):
        if self.fft_data is None or not self.vowel_models:
            return None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        #  cosinus cu fiecare fisier individual si cu mean vector
        cosine_distances = {}
        all_file_distances = {}
        avg_distances = {}
        combined_distances = {}
        
        for vowel, model in self.vowel_models.items():
            min_length = min(len(self.fft_data), len(model))
            mean_distance = cosine(self.fft_data[:min_length], model[:min_length])
            cosine_distances[vowel] = mean_distance
            
            #  toate fisierele pentru acest vowel
            vowel_dir = os.path.join(base_dir, vowel)
            all_file_distances[vowel] = []
            
            if os.path.exists(vowel_dir):
                sample_files = [f for f in os.listdir(vowel_dir) 
                              if f.endswith('.txt') and not f.startswith('mean_vector_')]
                
                print(f"\nDistante cosinus pentru vocala '{vowel}':")
                
                for sample_file in sample_files:
                    file_path = os.path.join(vowel_dir, sample_file)
                    try:
                        # Citire date
                        with open(file_path, 'r') as f:
                            data = []
                            for line in f:
                                try:
                                    value = float(line.strip())
                                    data.append(value)
                                except ValueError:
                                    pass
                        
                        if not data:
                            continue
                        
                        data = np.array(data[:1000])
                        data = data - np.mean(data)
                        if np.std(data) > 0:
                            data = data / np.std(data)
                        
                        fft_data = np.abs(np.fft.fft(data))
                        half_len = len(fft_data) // 2
                        
                        
                        fft_data = fft_data[:half_len]
                        
                        window_size = len(fft_data)
                        window = np.hamming(window_size)
                        fft_data = fft_data * window
                        
                        # normalizare pentru comparare uniforma
                        norm = np.linalg.norm(fft_data)
                        if norm > 0:
                            fft_data = fft_data / norm
                            
                        #  distanta cosinus
                        min_len = min(len(self.fft_data), len(fft_data))
                        distance = cosine(self.fft_data[:min_len], fft_data[:min_len])
                        all_file_distances[vowel].append(distance)
                        
                        print(f"  {sample_file}: {distance:.4f} (similaritate: {1-distance:.4f})")
                        
                    except Exception as e:
                        print(f"  Eroare la procesarea fisierului {sample_file}: {str(e)}")
                
                if all_file_distances[vowel]:
                    avg_distance = sum(all_file_distances[vowel]) / len(all_file_distances[vowel])
                    avg_distances[vowel] = avg_distance
                    
                    # 40% media individuala + 60% distanta cu mean vector
                    weight_avg = 0.4
                    weight_mean = 0.6
                    combined_dist = weight_avg * avg_distance + weight_mean * mean_distance
                    combined_distances[vowel] = combined_dist
                    
                    print(f"  Media distantelor pentru vocala {vowel}: {avg_distance:.4f} (similaritate: {1-avg_distance:.4f})")
                    print(f"  Distanta cu mean_vector_{vowel}: {mean_distance:.4f} (similaritate: {1-mean_distance:.4f})")
                    print(f"  Distanta combinata: {combined_dist:.4f} (similaritate: {1-combined_dist:.4f})")
        
        if combined_distances:
            recognized_vowel = min(combined_distances.items(), key=lambda x: x[1])[0]
        else:
            recognized_vowel = min(cosine_distances.items(), key=lambda x: x[1])[0]
        
        print("\nRezultate identificare vocala finala:")
        print("Distante cosinus cu mean vectors:")
        for vowel, dist in sorted(cosine_distances.items(), key=lambda x: x[1]):
            print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
        if avg_distances:
            print("\nMedia distantelor individuale:")
            for vowel, dist in sorted(avg_distances.items(), key=lambda x: x[1]):
                print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
        if combined_distances:
            print("\nDistante combinate (40% media individuala + 60% mean vector):")
            for vowel, dist in sorted(combined_distances.items(), key=lambda x: x[1]):
                print(f"Vocala {vowel}: {dist:.4f} (similaritate: {1-dist:.4f})")
            
            print(f"\nVocala recunoscuta: {recognized_vowel} (similaritate: {1-combined_distances[recognized_vowel]:.4f})")
        else:
            print(f"\nVocala recunoscuta: {recognized_vowel} (similaritate: {1-cosine_distances[recognized_vowel]:.4f})")
        

        display_distances = combined_distances if combined_distances else cosine_distances
        self.update_confidence_display(display_distances)
        
        
        return recognized_vowel
    
    def on_close(self):
        # inchiderea thread ului
        if self.reading_thread and self.reading_thread.is_alive():
            self.stop_reading = True
            self.reading_thread.join(timeout=0.5)
            
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MicSignal(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()


