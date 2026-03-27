import os

from pic import launch_gradio_app


if __name__ == "__main__":
    # Hugging Face Spaces / cloud-friendly defaults
    # Spaces usually provides PORT=7860; fall back to 7860.
    port = int(os.getenv("PORT", "7860"))
    launch_gradio_app(
        inbrowser=False,
        server_name="0.0.0.0",
        server_port=port,
    )

