
from IPython import display

def display_color(message, color = None):
    def color_fn():
        return display.HTML(f'<font style="color: {color}">{message}</font>')
    
    display(color_fn() if color else message)
