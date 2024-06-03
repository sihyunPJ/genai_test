import gradio as gr
from modules.gradio_ui import gradio_ui_tab
from modules.generators import t2i_generator, i2i_generator, ip_generator
import os
# os.chdir('.\\gradio')
# os.getcwd()


t2i_generator_tab, i2i_generator_tab, ip_generator_tab = gradio_ui_tab(t2i_generator, i2i_generator, ip_generator)

demo = gr.TabbedInterface([t2i_generator_tab, i2i_generator_tab, ip_generator_tab], ["Text to Image", "Image to Image", "Inpainting"])

if __name__ == "__main__":
    demo.launch(debug=True, share=True,server_port=8000)


