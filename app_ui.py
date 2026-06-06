import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MultimodalStreamUI(ctk.CTk):
    def __init__(self, toggle_callback):
        super().__init__()
        
        self.toggle_callback = toggle_callback
        self.get_frame_callback = None
        self.is_running = False 
        self.current_frame_img = None
        
        # CORREÇÃO: Cria a imagem offline fixa na memória UMA única vez
        blank_pil = Image.new("RGB", (640, 360), (40, 40, 40))
        self.offline_img = ctk.CTkImage(light_image=blank_pil, dark_image=blank_pil, size=(640, 360))
        
        self.title("FamilyFriend AI - Multimodal Pipeline")
        self.geometry("1100x700")
        self.minsize(1100, 700)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._build_sidebar()
        self._build_main_panel()
        
        self.update_video_feed()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🛡️ FamilyFriend", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))
        
        self.status_badge = ctk.CTkLabel(self.sidebar, text="● INATIVO", text_color="gray", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_badge.grid(row=1, column=0, padx=20, pady=10)
        
        self.btn_toggle = ctk.CTkButton(self.sidebar, text="INICIAR PIPELINE", height=40, fg_color="#10B981", hover_color="#059669", font=ctk.CTkFont(weight="bold"), command=self.on_toggle_clicked)
        self.btn_toggle.grid(row=2, column=0, padx=20, pady=20)
        
        self.switch_realtime = ctk.CTkSwitch(self.sidebar, text="Som ao Vivo (Loopback)")
        self.switch_realtime.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")
        self.switch_realtime.deselect() 
        
        self.switch_logs = ctk.CTkSwitch(self.sidebar, text="Exibir Logs (Console)")
        self.switch_logs.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="w")
        self.switch_logs.select() 
        
        self.lbl_thresh = ctk.CTkLabel(self.sidebar, text="Sensibilidade: 0.50", font=ctk.CTkFont(size=12))
        self.lbl_thresh.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.slider_thresh = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, command=self._update_thresh_lbl)
        self.slider_thresh.set(0.5)
        self.slider_thresh.grid(row=6, column=0, padx=20, pady=(10, 20))

    def _build_main_panel(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.dash_frame = ctk.CTkFrame(self.main_frame, height=80)
        self.dash_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        self.dash_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.rtf_box = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        self.rtf_box.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkLabel(self.rtf_box, text="REAL TIME FACTOR (RTF)", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.rtf_label = ctk.CTkLabel(self.rtf_box, text="--", font=ctk.CTkFont(size=24, weight="bold"))
        self.rtf_label.pack(anchor="w")
        
        self.blocks_box = ctk.CTkFrame(self.dash_frame, fg_color="transparent")
        self.blocks_box.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        ctk.CTkLabel(self.blocks_box, text="OFENSAS BLOQUEADAS", font=ctk.CTkFont(size=11)).pack(anchor="e")
        self.blocks_label = ctk.CTkLabel(self.blocks_box, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color="#EF4444")
        self.blocks_label.pack(anchor="e")
        
        self.video_frame = ctk.CTkFrame(self.main_frame, height=360)
        self.video_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="SINAL DE VÍDEO OFFLINE", font=ctk.CTkFont(size=20), image=self.offline_img)
        self.video_label.grid(row=0, column=0)
        
        self.log_console = ctk.CTkTextbox(self.main_frame, height=120, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_console.grid(row=2, column=0, pady=(10,0), sticky="ew")
        self.log_console.insert("0.0", "Sistema inicializado. Tela e Áudio prontos.\n")
        self.log_console.configure(state="disabled")

    def get_threshold(self): return self.slider_thresh.get()
    def is_realtime_enabled(self): return self.switch_realtime.get() == 1

    def _update_thresh_lbl(self, value):
        self.lbl_thresh.configure(text=f"Sensibilidade: {value:.2f}")

    def safe_ui_update(self, msg=None, rtf_val=None, total_censored=None):
        self.after(0, self._apply_ui_updates, msg, rtf_val, total_censored)

    def _apply_ui_updates(self, msg, rtf_val, total_censored):
        # Mantenha o log de texto opcional
        if msg and self.switch_logs.get() == 1:
            self.log_console.configure(state="normal")
            self.log_console.insert("end", msg + "\n")
            self.log_console.see("end")
            self.log_console.configure(state="disabled")
            
        # O RTF e total_censored DEVEM ser atualizados sempre que o backend enviar
        if rtf_val is not None:
            color = "#10B981" if rtf_val <= 1.0 else "#EF4444"
            self.rtf_label.configure(text=f"{rtf_val:.3f}x", text_color=color)
            
        if total_censored is not None:
            self.blocks_label.configure(text=str(total_censored))
    def update_video_feed(self):
        if self.is_running and self.get_frame_callback:
            frame = self.get_frame_callback()
            if frame is not None:
                pil_img = Image.fromarray(frame)
                self.current_frame_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 360))
                self.video_label.configure(image=self.current_frame_img, text="")
                
        self.after(30, self.update_video_feed)

    def on_toggle_clicked(self):
        if not self.is_running:
            self.is_running = True
            self.status_badge.configure(text="● GRAVANDO", text_color="#EF4444")
            self.btn_toggle.configure(text="PARAR E SALVAR", fg_color="#EF4444", hover_color="#B91C1C")
            self.slider_thresh.configure(state="disabled")
            self.blocks_label.configure(text="0")
            self.rtf_label.configure(text="--", text_color="white")
            self.safe_ui_update(msg="\n" + "="*40 + "\n🚀 INICIANDO PIPELINE MULTIMODAL...\n" + "="*40)
            self.toggle_callback(is_starting=True)
        else:
            self.is_running = False
            self.status_badge.configure(text="● INATIVO", text_color="gray")
            self.btn_toggle.configure(text="INICIAR PIPELINE", fg_color="#10B981", hover_color="#059669")
            self.slider_thresh.configure(state="normal")
            
            # CORREÇÃO: Resgata a imagem fixa da memória de forma segura
            self.current_frame_img = None
            self.video_label.configure(image=self.offline_img, text="SINAL DE VÍDEO OFFLINE")
            
            self.safe_ui_update(msg="\n🛑 PIPELINE INTERROMPIDA PELO USUÁRIO.")
            self.toggle_callback(is_starting=False)