import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig = plt.figure()
ax = plt.axes()

image = np.loadtxt("data/gd_images/gd_image_0.txt", skiprows=5)

im = ax.imshow(image, cmap="viridis")

def init():
    im.set_data(np.zeros(image.shape))
    return

def animate(n):
    
    image = np.loadtxt(f"data/gd_images/gd_image_{n}.txt", skiprows=5)
    im.set_data(image)                 
    return

ani = animation.FuncAnimation(fig, animate, init_func=init, frames = 50,
                              interval = 100, blit = False, repeat = False)

plt.show()
