### Some of the visualization code in this file is copied and edited from that of Gaze360.
### You can see the original code by accessing "Updated online demo" on https://github.com/erkil1452/gaze360

from matplotlib.pyplot import imshow, show
from OpenGL.GL import shaders
import glfw
import numpy as np
import OpenGL.GL as gl
from pathlib import Path
import cv2
from math import sqrt

WIDTH, HEIGHT = 960, 720
glfw.init()
window = glfw.create_window(WIDTH, HEIGHT, "Draw Arrow Window", None, None)
if not window:
    glfw.terminate()
    print("Failed to create window")
    exit()

glfw.make_context_current(window)

vertexPositions = np.float32([[-1, -1], [1, -1], [-1, 1], [1, 1]])
with (Path(__file__).parent / "shader/vertex_shader.glsl").open("r") as f:
    VERTEX_SHADER = shaders.compileShader(
        f.read(),
        gl.GL_VERTEX_SHADER,
    )


with (Path(__file__).parent / "shader/fragment_shader.glsl").open("r") as f:
    FRAGMENT_SHADER = shaders.compileShader(
        f.read(),
        gl.GL_FRAGMENT_SHADER,
    )

shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

xpos = gl.glGetUniformLocation(shader, "xpos")
ypos = gl.glGetUniformLocation(shader, "ypos")

vdir_x = gl.glGetUniformLocation(shader, "vdir_x")
vdir_y = gl.glGetUniformLocation(shader, "vdir_y")
vdir_z = gl.glGetUniformLocation(shader, "vdir_z")

arrow_color = gl.glGetUniformLocation(shader, "arrow_color")

arrow_size = gl.glGetUniformLocation(shader, "size")

arrow_length_mul = gl.glGetUniformLocation(shader, "arrow_length_mul")

res_loc = gl.glGetUniformLocation(shader, "iResolution")


def render_frame(
    x_position: float,
    y_position: float,
    vx: float,
    vy: float,
    vz: float,
    box_position: tuple[float, float],
    box_size: tuple [float,float],
    object_class_name: str,
    acolor: tuple[float, float, float] = (1.0, 0.0, 0.0),
    asize: float = 0.05,
    alength: float = 1.0,
    offset: float = 0.03,
    color=(0, 255, 0), 
    thickness=2
) -> np.ndarray:
    """
    Draw arrow on image sized (HEIGHT,WIDTH) and returns it.
    Params:
        x_position: x coordinate of arrow root in [-1,1]; right is positive
        y_position: y coordinate of arrow root in [-1,1]; up is positive
        vx: horizontal component of arrow direction. right is positive
        vy: vertical component of arrow direction. up is positive
        vz: depth component of arrow direction. near side is positive
        box_position: the center of the bounding box for the object
        box_size: the size (width, height) of the bounding box
        acolor: RGB color
        asize: up to 0.05 is recommended
        alength: arrow length
        offset: How much the arrow is displaced to the arrow direction
        color (tuple): BGR color values for the bounding box.
        thickness (int): The thickness of the bounding box lines.
    """

    from demo_object_detection import class_labels
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    with shader:
        x_position += vx * offset * (asize / 0.05)
        y_position += vy * offset * (asize / 0.05)

        x_position = x_position * 0.89
        y_position = y_position * 0.67
        gl.glUniform1f(xpos, x_position)
        gl.glUniform1f(ypos, y_position)

        gl.glUniform1f(vdir_x, vx)
        gl.glUniform1f(vdir_y, vy)
        gl.glUniform1f(vdir_z, vz)
        gl.glUniform1f(arrow_size, asize)
        gl.glUniform1f(arrow_length_mul, alength)

        gl.glUniform3f(res_loc, WIDTH, HEIGHT, 1.0)

        gl.glUniform3f(arrow_color, acolor[0], acolor[1], acolor[2])

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, vertexPositions)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]

    """ my added code"""
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Calculate the top-left corner of the bounding box
    top_left = (int(box_position[0] - box_size[0] / 2), int(box_position[1] - box_size[1] / 2))
    # Calculate the bottom-right corner of the bounding box
    bottom_right = (int(box_position[0] + box_size[0] / 2), int(box_position[1] + box_size[1] / 2))
    # Draw the rectangle on the image
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    label_position = (top_left[0], top_left[1] - 10)
    cv2.putText(img, object_class_name , label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    """end of my added code"""



    return img

""" below is the code snipped that draws a box around the object that the person is pointing to.
    Just for testing"""

def draw_bounding_box(image, position, box_size, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box around an object given its position.

    Parameters:
        image (np.ndarray): The image where the bounding box will be drawn.
        position (tuple): The center position (x, y) of the bounding box in image coordinates.
        box_size (tuple): The size (width, height) of the bounding box.
        color (tuple): BGR color values for the bounding box.
        thickness (int): The thickness of the bounding box lines.

    Returns:
        np.ndarray: The image with the bounding box drawn.
    """
    # Calculate the top-left corner of the bounding box
    top_left = (int(position[0] - box_size[0] / 2), int(position[1] - box_size[1] / 2))
    # Calculate the bottom-right corner of the bounding box
    bottom_right = (int(position[0] + box_size[0] / 2), int(position[1] + box_size[1] / 2))
    # Draw the rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    return image

# For debugging.
if __name__ == "__main__":
    import cv2
    from math import sqrt

    x = 1
    y = 1
    z = 5
    r = sqrt(x**2 + y**2 + z**2)

    img_arrow = (
        render_frame(
            0.8,
            0,
            x / r,
            y / r,
            z / r,
            box_position = (350, 300),
            box_size= (150, 100),
            acolor=(1, 0, 0),
            asize=0.3,
            offset=0,
        )
        / 255
    )

    cv2.imwrite("arrow.png", img_arrow * 255)

    arrow_mask = (
        (img_arrow[:, :, 0] + img_arrow[:, :, 1] + img_arrow[:, :, 2]) == 0.0
    ).astype(float)[:, :, None]
    img_arrow_alpha = np.concatenate((img_arrow, 1 - arrow_mask), axis=2)
    cv2.imwrite("arrow_alpha.png", img_arrow_alpha * 255)
