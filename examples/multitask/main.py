from FlashRL import Game


def on_frame(state, frame, type, vnc):
    pass




Game("multitask", fps=10, frame_callback=on_frame, grayscale=True, normalized=True)
