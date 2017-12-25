from FlashRL import Game


def on_frame(state, type, vnc):
    print(state.shape, type)

Game("mujaffa-v1.6", fps=1, frame_callback=on_frame, grayscale=True, normalized=True)