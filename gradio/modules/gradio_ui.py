import gradio as gr

from modules.generators import t2i_generator, i2i_generator, ip_generator


# t2i_generator = t2i_generator()
# i2i_generator = i2i_generator()
# ip_generator = ip_generator()

def gradio_ui_tab (t2i_generator, i2i_generator, ip_generator):
    t2i_generator_tab = gr.Interface(fn=t2i_generator,
                                    inputs=["textbox",
                                            "textbox",
                                            gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
                                            gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
                                            gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
                                            gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
                                            gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
                                            "checkbox"
                                            ]
                        , outputs=["text", "image"]
                        , title = "Innople AI Studio")

    i2i_generator_tab = gr.Interface(fn=i2i_generator,
                                    inputs=["textbox",
                                            "textbox",
                                            gr.Image(value=None, type="pil"),
                                            gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
                                            gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
                                            gr.Slider(0, 1, step=0.01, value=0.75, label="Denoising Strength", show_label=True ),
                                            gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
                                            gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
                                            gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
                                            "checkbox"
                                            ]
                        , outputs=["text", "image"]
                        , title = "Innople AI Studio")

    ip_generator_tab = gr.Interface(fn=ip_generator,
                                    inputs=["textbox",
                                            "textbox",
                                            gr.Image(value=None, type="pil"),
                                            gr.Image(value=None, type="pil"),
                                            gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
                                            gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
                                            gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
                                            gr.Slider(0, 1, step=0.01, value=0.75, label="Denoising Strength", show_label=True ),
                                            gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
                                            gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
                                            gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
                                            "checkbox"
                                            ]
                        , outputs=["text", "image"]
                        , title = "Innople AI Studio")
    return t2i_generator_tab, i2i_generator_tab, ip_generator_tab