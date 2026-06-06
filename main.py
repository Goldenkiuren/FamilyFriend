from app_ui import MultimodalStreamUI
from core_pipeline import MultimodalPipeline

def main():
    # 1. Instanciamos o Backend
    pipeline = MultimodalPipeline()

    # 2. Definimos a função que o botão da UI vai chamar
    def on_toggle(is_starting):
        if is_starting:
            pipeline.start()
        else:
            pipeline.stop()

    # 3. Instanciamos a UI
    app = MultimodalStreamUI(toggle_callback=on_toggle)

    # 4. Injeção de Dependências
    pipeline.ui_update = app.safe_ui_update
    pipeline.get_threshold = app.get_threshold
    pipeline.is_realtime = app.is_realtime_enabled
    
    # NOVIDADE: Ensinamos a UI como ir buscar o vídeo ao Backend
    app.get_frame_callback = pipeline.get_display_frame

    # 5. Tratamento de encerramento
    def on_closing():
        pipeline.stop()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Inicia a interface
    app.mainloop()

if __name__ == "__main__":
    main()