from PIL import Image
import imageio
import glob
import numpy as np

frames = []

SCALE_UP = 10

for i in range(74, 0, -1):
    frame = Image.open(f"./results/{i}timestep_diffused.png")
    frame = np.array(frame)

    a = frame[:, :, 0]
    b = frame[:, :, 1]
    c = frame[:, :, 2]

    a = np.kron(a, np.ones((SCALE_UP, SCALE_UP)))
    b = np.kron(b, np.ones((SCALE_UP, SCALE_UP)))
    c = np.kron(c, np.ones((SCALE_UP, SCALE_UP)))

    a = np.expand_dims(a, axis=-1)
    b = np.expand_dims(b, axis=-1)
    c = np.expand_dims(c, axis=-1)

    frame = np.concatenate((a, b, c), axis=2)

    frame = frame.astype(np.uint8)

    image = Image.fromarray(frame)
    image.save(f"./results/{i}timestep_diffused.png")

    frames.append(imageio.imread(f"./results/{i}timestep_diffused.png"))

for i in range(40):
    frames.append(imageio.imread(f"./results/{1}timestep_diffused.png"))

imageio.mimsave('diffusion4.gif', frames, duration=2.0, loop=0)
