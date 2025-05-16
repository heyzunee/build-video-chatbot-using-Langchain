import gradio as gr
from src.handle import Handler

if __name__ == "__main__":
    handler = Handler()
    with open("src/template/style.css", "r") as f:
        css = f.read()

    # Create the Gradio interface
    with gr.Blocks(
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.red,
            secondary_hue=gr.themes.colors.pink,
            font=[
                gr.themes.GoogleFont("Montserrat"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
        )
    ) as demo:

        gr.HTML("<style>" + css + "</style>")

        video_chunks = gr.State(None)
        # has_processed_video = gr.State(False)

        with gr.Row(elem_id="main-container"):
            # --- Sidebar Upload Panel ---
            with gr.Column(scale=1, min_width=320, elem_id="upload-container"):
                gr.Markdown("## 🎬 Upload Video")

                video_file = gr.File(
                    label="Upload Video",
                    show_label=False,
                    file_types=["video"],
                    elem_classes="upload-container__file",
                )

                gr.Markdown("### 🔗 Or paste a YouTube link")

                video_link = gr.Textbox(
                    label="YouTube Link",
                    show_label=False,
                    placeholder="https://www.youtube.com/watch?v=...",
                    elem_classes="upload-container__link",
                )

                upload_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    elem_classes="upload-container__button",
                )

                with gr.Column():
                    status_output = gr.Markdown(
                        "⌛ Waiting for video to be processed...",
                        elem_classes="upload-container__status",
                    )

            # --- Main Chat Area ---
            with gr.Column(scale=3, elem_id="chat-container"):
                gr.Markdown("# 🤖 YouBot - Your video assistant")
                gr.Markdown("### Welcome to YouBot!")
                gr.Markdown(
                    "Upload a video or provide a YouTube link from the left panel to start asking questions about its content."
                )

                with gr.Column(elem_id="chat-interface"):
                    # gr.ChatInterface(
                    #     fn=handle,
                    #     type="messages",
                    #     examples=["What is the capital of France?", "Is the sky blue?"],
                    # )

                    with gr.Row():
                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            bubble_full_width=False,
                            type="messages",
                            height=500,
                            show_label=False,
                        )

                    with gr.Row():
                        chat_input = gr.MultimodalTextbox(
                            placeholder="Ask anything...",
                            show_label=False,
                            interactive=False,
                            file_count="multiple",
                            sources=["microphone", "upload"],
                            file_types=["image"],
                            elem_classes="chat-interface__input",
                        )

                        chat_input.submit(
                            handler.submit_message,
                            inputs=[chat_input, chatbot],
                            outputs=[chat_input, chatbot],
                        )

                        chatbot.like(
                            handler.print_like_dislike,
                            None,
                            None,
                            like_user_message=True,
                        )

        # Handle video upload
        upload_btn.click(
            fn=handler.upload_video,
            inputs=[video_file, video_link],
            outputs=[video_chunks, chat_input, status_output],
        )

    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", server_port=8501)
