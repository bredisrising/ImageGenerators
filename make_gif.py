from PIL import Image
import imageio
import glob
import numpy as np

frames = []


for i in range(73, 0, -1):
    frame = Image.open(f"./results/{i}timestep_diffused.png")
    frame = np.array(frame)

    a = frame[:, :, 0]
    b = frame[:, :, 1]
    c = frame[:, :, 2]

    a = np.kron(a, np.ones((20, 20)))
    b = np.kron(b, np.ones((20, 20)))
    c = np.kron(c, np.ones((20, 20)))

    a = np.expand_dims(a, axis=-1)
    b = np.expand_dims(b, axis=-1)
    c = np.expand_dims(c, axis=-1)

    frame = np.concatenate((a, b, c), axis=2)

    frame = frame.astype(np.uint8)

    image = Image.fromarray(frame)
    image.save(f"./results/{i}timestep_diffused.png")

    frames.append(imageio.imread(f"./results/{i}timestep_diffused.png"))

imageio.mimsave('diffusion.gif', frames, duration=1.0, loop=0)
