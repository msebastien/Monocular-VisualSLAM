import OpenGL.GL as gl


def read_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


def parse_camera_intrinsics(path, camera_id="P0"):
    param_lines = read_file(path)

    f, cx, cy = 0
    for line in param_lines:
        # Ignore comments
        if line.startswith("#"):
            break
        
        if line.startswith(camera_id):
            # Split the line and convert to float
            values = line.strip().split()

            match values[0]:
                case "f":
                    f = float(values[1])
                case "cx":
                    cx = float(values[1])
                case "cy":
                    cy = float(values[1])
                case "rx":
                    cx = float(values[1]) // 2
                case "ry":
                    cy = float(values[1]) // 2
                case _:
                    print("Unknown camera intrinsic parameter")

    return f, cx, cy


def draw_points(points):
    """
    points: an array of point. Each point = 3 components
    """
    gl.glBegin(gl.GL_POINTS)

    for p in points:
        gl.glVertex3d(p[0], p[1], p[2])

    gl.glEnd()


def draw_points_colored(points, colors):
    """
    - points: an array of point. Each point = 3 components
    - colors: an array of color. Each color = 3 components
    """
    gl.glBegin(gl.GL_POINTS)

    for p, c in zip(points, colors):
        gl.glColor3f(c[0], c[1], c[2])
        gl.glVertex3f(p[0], p[1], p[2])

    gl.glEnd()


def draw_cameras(cameras, w=1.0, h_ratio=0.75, z_ratio=0.6):
    """
    cameras: an array of camera pose matrix (4x4)
    """
    h = w * h_ratio
    z = w * z_ratio

    for cam in cameras:
        gl.glPushMatrix()
        gl.glMultTransposeMatrixd(cam)

        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(w, h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(w, -h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-w, h, z)

        gl.glVertex3f(w, h, z)
        gl.glVertex3f(w, -h, z)

        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(-w, -h, z)

        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(w, h, z)

        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(w, -h, z)
        gl.glEnd()

        gl.glPopMatrix()


def draw_camera(camera, w=1.0, h_ratio=0.75, z_ratio=0.6):
    """
    camera: camera pose matrix (4x4)
    """
    h = w * h_ratio
    z = w * z_ratio

    gl.glPushMatrix()
    gl.glMultTransposeMatrixd(camera)

    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(w, h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(w, -h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(-w, -h, z)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(-w, h, z)

    gl.glVertex3f(w, h, z)
    gl.glVertex3f(w, -h, z)

    gl.glVertex3f(-w, h, z)
    gl.glVertex3f(-w, -h, z)

    gl.glVertex3f(-w, h, z)
    gl.glVertex3f(w, h, z)

    gl.glVertex3f(-w, -h, z)
    gl.glVertex3f(w, -h, z)
    gl.glEnd()

    gl.glPopMatrix()


def draw_line(points, point_size=0):
    """
    points: an array of point. Each point = 3 components
    """
    gl.glBegin(gl.GL_LINES)
    for i, p in enumerate(points):
        gl.glVertex3d(points[i][0], points[i][1], points[i][2])
        gl.glVertex3d(points[i + 1][0], points[i + 1][1], points[i + 1][2])
    gl.glEnd()

    if point_size > 0:
        gl.glPointSize(point_size)
        gl.glBegin(gl.GL_POINTS)
        for p in points:
            gl.glVertex3d(p[0], p[1], p[2])
        gl.glEnd()


def draw_lines(points, point_size=0):
    """
    - points: an array of point. Each point = 3 components.
    - Here, each row in the array has 2 points, or 6 components in total
    """

    gl.glBegin(gl.GL_LINES)
    for p in points:
        gl.glVertex3d(p[0], p[1], p[2])
        gl.glVertex3d(p[3], p[4], p[5])
    gl.glEnd()

    if point_size > 0:
        gl.glPointSize(point_size)
        gl.glBegin(gl.GL_POINTS)
        for p in points:
            gl.glVertex3d(p[0], p[1], p[2])
            gl.glVertex3d(p[3], p[4], p[5])
        gl.glEnd()


def draw_lines2(points, points2, point_size=0):
    """
    - points: an array of point.
    - points2: another array of point.
    - Trace a line between a point in the first array
    and a point in the second array
    """
    size = min(points.shape[0], points2.shape[0])

    gl.glBegin(gl.GL_LINES)
    for i in range(0, size):
        gl.glVertex3d(points[i][0], points[i][1], points[i][2])
        gl.glVertex3d(points2[i][0], points2[i][1], points2[i][2])
    gl.glEnd()

    if point_size > 0:
        gl.glPointSize(point_size)
        gl.glBegin(gl.GL_POINTS)
        for i in range(0, size):
            gl.glVertex3d(points[i][0], points[i][1], points[i][2])
            gl.glVertex3d(points2[i][0], points2[i][1], points2[i][2])
        gl.glEnd()


def draw_boxes(cameras, sizes):
    """
    - cameras: an array of camera pose matrix (4x4)
    - sizes: an array of size for each component (3D)
    """
    for cam, size in zip(cameras, sizes):
        gl.glPushMatrix()
        gl.glMultTransposeMatrixd(cam)

        w = size[0] / 2.0  # w/2
        h = size[1] / 2.0
        z = size[2] / 2.0

        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(-w, -h, -z)
        gl.glVertex3f(w, -h, -z)
        gl.glVertex3f(-w, -h, -z)
        gl.glVertex3f(-w, h, -z)
        gl.glVertex3f(-w, -h, -z)
        gl.glVertex3f(-w, -h, z)

        gl.glVertex3f(w, h, -z)
        gl.glVertex3f(-w, h, -z)
        gl.glVertex3f(w, h, -z)
        gl.glVertex3f(w, -h, -z)
        gl.glVertex3f(w, h, -z)
        gl.glVertex3f(w, h, z)

        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(w, h, z)
        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(-w, h, -z)

        gl.glVertex3f(w, -h, z)
        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(w, -h, z)
        gl.glVertex3f(w, h, z)
        gl.glVertex3f(w, -h, z)
        gl.glVertex3f(w, -h, -z)
        gl.glEnd()

        gl.glPopMatrix()
