from multiprocessing import Process, Queue
import numpy as np

import pypangolin as pango
import OpenGL.GL as gl


# Global map // 3D visualization using pangolin
class Map(object):
    def __init__(self):
        self.frames = []  # camera frames (means camera pose)
        self.points = []  # 3D points of map
        self.state = None  # holds current state of the map and cam pose
        # A queue for inter-process communication (for visualization process)
        self.q = None

    def create_viewer(self):
        # Parallel Execution: The main purpose of creating this process is to run
        # the `viewer_thread` method in parallel with the main program.
        # This allows the 3D viewer to update and render frames continuously
        # without blocking the main execution flow.
        self.q = Queue()

        # initializes the Parallel process with the `viewer_thread` function
        # the arguments that the function takes is mentioned in the args var
        p = Process(target=self.viewer_thread, args=(self.q,))

        # daemon true means, exit when main program stops
        p.daemon = True

        # starts the process
        p.start()

    def viewer_thread(self, queue):
        # The viewer thread takes the queue as input
        # Initializes the vizualisation window
        self.viewer_init(1280, 720)
        # An infinite loop that continually refreshes the viewer
        while True:
            self.viewer_refresh(queue)

    def viewer_init(self, width, height):
        pango.CreateWindowAndBind("Main", width, height)

        # This ensures that only the nearest objects are rendered,
        # creating a realistic representation of the scene with
        # correct occlusions
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Sets up the camera with a projection matrix and a model-view matrix
        self.scam = pango.OpenGlRenderState(
            #  'ProjectionMatrix' The parameters specify the viewport width/height,
            #  the focal lengths in the x and y directions (420, 420),
            #  the principal point coordinates (w//2, h//2),
            #  and the near and far clipping planes (0.2, 10000).
            #  The focal lengths determine the field of view,
            #  the principal point indicates the center of the projection
            #  and the clipping planes define the range of distances from the camera
            #  within which objects are rendered, with objects closer than
            #  0.2 units or farther than 10000 uits being clipped out of the scene.
            pango.ProjectionMatrix(
                width, height, 420, 420, width // 2, height // 2, 0.2, 10000
            ),
            # 'ModelViewLookAt' sets up the camera view matrix, which
            #  defines the position and orientation of the camera in the 3D
            #  scene. The first three parameters (0, -10, -8) specify the position
            #  of the camera in the world coordinates, indicating that the camera
            #  is located at coordinates (0, -10, -8). The next three parameters
            #  (0, 0, 0) define the point in space the camera is looking at,
            #  which is the origin in this case.
            #  The last three parameters (0, -1, 0) represent the up direction vector
            #  indicating which direction is considered 'up' for the camera,
            #  here pointing along the negative y-axis.
            #  This setup effectively positions the camera 10 units down and
            #  8 units back from the origin, looking towards the origin with
            #  the 'up' direction being downwards in the y-axis, which is
            #  unconventional and might be used to achieve a specific
            #  orientation or perspective in the rendered scene.
            pango.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0)
        )
        # Creates a handler for 3D interaction
        self.handler = pango.Handler3D(self.scam)

        # Creates a display context
        self.dcam = pango.CreateDisplay()
        # Sets the bounds of the display
        self.dcam.SetBounds
        (
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach(0),
            pango.Attach(1),
            float(-width) / float(height)
        )
        # Assigns handler for mouse clicking and stuff, interactive
        self.dcam.SetHandler(self.handler)
        self.darr = None

    def viewer_refresh(self, queue):
        #  Checks of the current state is None or if the queue is not empty
        if self.state is None or not queue.empty():
            #  Gets the latest state from the queue
            self.state = queue.get()

        #  Clears the color and depth buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        #  Sets the clear color to white
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        #  Activates the display context with the current camera
        self.dcam.Activate(self.scam)

        #  Camera trajectory line and color setup
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        pango.DrawCameras(self.state[0])

        #  3D point cloud color setup
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pango.DrawPoints(self.state[1])

        #  Finishes the current frame and swaps the buffers.
        pango.FinishFrame()

    def display(self):
        if self.queue is None:
            return

        poses, pts = [], []
        for f in self.frames:
            #  Updating pose
            poses.append(f.pose)

        for p in self.points:
            #  Updating map points
            pts.append(p.pt)

        #  Updating the queue
        self.queue.put((np.array(poses), np.array(pts)))


class Point(object):
    #  A Point is a 3D point in the world
    #  Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        # The Point initializes with a location and unique ID,
        # adds itself to the map’s list of points,
        # and records observations with associated frames and feature indices.
        self.frames = []
        self.pt = loc
        self.idxs = []

        #  Assigns a unique ID to the point based on the current number of points in the map
        self.id = len(mapp.points)
        #  Adds the point instance to the map's list of points.
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)
