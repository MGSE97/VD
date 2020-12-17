import glob
import os
import sys
import numpy as np
from multiprocessing import Process, Queue
from vispy import app, scene

save_dir = "./sph_17_40x40x40_mu_3_5_iso_81_2/"
q = Queue()


def load(q):
    i = 0
    saved_files = list(filter(os.path.isfile, glob.glob(save_dir + "*.npy")))
    size = len(saved_files)
    cache = {}
    while True:
        if i >= size:
            i = 0

        key = str(i)
        if key not in cache:
            cache[key] = np.load(saved_files[i])
            print(os.path.basename(saved_files[i])[:-4], end=", ")
        if len(cache[key]) > 0:
            q.put(cache[key])

        break
        i += 1


surface = []
view = None
def render(q):
    global surface
    global view

    # Create a canvas with a 3D viewport
    canvas = scene.SceneCanvas(keys='interactive')
    view = canvas.central_widget.add_view()

    # Create mesh visual
    data = np.load(save_dir + "sph_000000.npy")
    surface = scene.visuals.Mesh(data, color=(0.5, 0.6, 1, 1), shading='smooth', parent=view.scene)
    surface.transform = scene.transforms.STTransform(translate=(0.5, 0.5, 0))

    def redraw(event):
        global surface
        global view

        if q.empty():
            pass
        else:
            #surface.parent = None
            data = q.get()
            #surface = scene.visuals.Mesh(data, color=(0.5, 0.6, 1, 1), shading='smooth', parent=view.scene)
            surface.set_data(data, color=(0.5, 0.6, 1, 1))
            #print(surface.mesh_data.get_vertex_normals(indexed='faces').size, surface.mesh_data.get_vertices(indexed='faces').size)
            #surface._normals._size = surface._vertices.size
            #surface.shared_program.vert['normal'] = surface.mesh_data.get_vertex_normals(indexed='faces')
            #surface.transform = scene.transforms.STTransform(translate=(0.5, 0.5, 0))
            #surface._normals.set_data(np.ones((0, 3), dtype=np.float32))#surface.mesh_data._vertex_normals
            #surface.update()
            canvas.update()
            #canvas.on_draw(None)

    timer = app.timer.Timer()
    timer.connect(redraw)
    timer.start(0.1)

    # Add a 3D axis
    axis = scene.visuals.XYZAxis(parent=view.scene)

    # Use a 3D camera
    cam = scene.TurntableCamera(elevation=30, azimuth=30)
    cam.set_range((0, 1), (0, 1), (0, 1))
    view.camera = cam

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()


if __name__ == '__main__':
    p2 = Process(target=load, args=[q])
    p1 = Process(target=render, args=[q])
    p1.start()
    p2.start()
    p2.join()
    p1.join()
