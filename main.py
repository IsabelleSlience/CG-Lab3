import argparse
from pathlib import Path

import numpy as np
import taichi as ti
from PIL import Image

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100
CURVE_POINT_COUNT = NUM_SEGMENTS + 1
LINE_VERTEX_COUNT = 2 * (MAX_CONTROL_POINTS - 1)

BACKGROUND_COLOR = (1.0, 1.0, 1.0)
CURVE_COLOR = (0.12, 0.72, 0.22)
CONTROL_POINT_COLOR = (0.9, 0.2, 0.2)
CONTROL_POLYGON_COLOR = (0.55, 0.55, 0.55)
CONTROL_POINT_RADIUS = 0.008
HIDDEN_POINT = -10.0


def init_taichi():
    try:
        ti.init(arch=ti.gpu)
        return "gpu"
    except Exception:
        ti.init(arch=ti.cpu)
        return "cpu"


ACTIVE_ARCH = init_taichi()

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=CURVE_POINT_COUNT)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_radii = ti.field(dtype=ti.f32, shape=MAX_CONTROL_POINTS)
line_vertices = ti.Vector.field(2, dtype=ti.f32, shape=LINE_VERTEX_COUNT)


def de_casteljau(points, t):
    working = np.asarray(points, dtype=np.float32)
    if working.ndim != 2 or working.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if working.shape[0] == 0:
        raise ValueError("at least one control point is required")

    while working.shape[0] > 1:
        working = (1.0 - t) * working[:-1] + t * working[1:]
    return working[0]


def sample_bezier(points, num_segments):
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float32)

    samples = np.empty((num_segments + 1, 2), dtype=np.float32)
    for i in range(num_segments + 1):
        t = i / num_segments
        samples[i] = de_casteljau(points, t)
    return samples


def build_gui_points_array(control_points, max_points):
    gui_array = np.full((max_points, 2), HIDDEN_POINT, dtype=np.float32)
    if control_points:
        points = np.asarray(control_points[:max_points], dtype=np.float32)
        gui_array[: len(points)] = points
    return gui_array


def build_gui_radii_array(control_points, max_points, radius=CONTROL_POINT_RADIUS):
    radii = np.zeros(max_points, dtype=np.float32)
    radii[: min(len(control_points), max_points)] = radius
    return radii


def build_line_vertices_array(control_points, max_points):
    vertices = np.full((2 * (max_points - 1), 2), HIDDEN_POINT, dtype=np.float32)
    if len(control_points) < 2:
        return vertices

    capped = np.asarray(control_points[:max_points], dtype=np.float32)
    pair_count = len(capped) - 1
    for i in range(pair_count):
        vertices[2 * i] = capped[i]
        vertices[2 * i + 1] = capped[i + 1]
    return vertices


@ti.kernel
def clear_pixels_kernel():
    for x, y in pixels:
        pixels[x, y] = ti.Vector(BACKGROUND_COLOR)


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        point = curve_points_field[i]
        px = ti.cast(point[0] * (WIDTH - 1), ti.i32)
        py = ti.cast(point[1] * (HEIGHT - 1), ti.i32)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = ti.Vector(CURVE_COLOR)


def update_gpu_buffers(control_points):
    gui_points.from_numpy(build_gui_points_array(control_points, MAX_CONTROL_POINTS))
    gui_radii.from_numpy(build_gui_radii_array(control_points, MAX_CONTROL_POINTS))
    line_vertices.from_numpy(build_line_vertices_array(control_points, MAX_CONTROL_POINTS))

    if len(control_points) >= 2:
        sampled_curve = sample_bezier(control_points, NUM_SEGMENTS)
        curve_points_field.from_numpy(sampled_curve)
        draw_curve_kernel(CURVE_POINT_COUNT)


def render_frame(control_points):
    clear_pixels_kernel()
    update_gpu_buffers(control_points)


def normalized_to_pixel(point):
    x = int(np.clip(point[0], 0.0, 1.0) * (WIDTH - 1))
    y = int(np.clip(point[1], 0.0, 1.0) * (HEIGHT - 1))
    return x, y


def draw_disk_cpu(image, center, radius, color):
    cx, cy = normalized_to_pixel(center)
    radius_sq = radius * radius
    x0 = max(0, cx - radius)
    x1 = min(WIDTH - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(HEIGHT - 1, cy + radius)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= radius_sq:
                image[y, x] = color


def draw_line_cpu(image, start, end, color):
    x0, y0 = normalized_to_pixel(start)
    x1, y1 = normalized_to_pixel(end)
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        alpha = i / steps
        x = int(round((1.0 - alpha) * x0 + alpha * x1))
        y = int(round((1.0 - alpha) * y0 + alpha * y1))
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            image[y, x] = color


def build_preview_image(control_points):
    image = np.ones((HEIGHT, WIDTH, 3), dtype=np.float32)
    if len(control_points) >= 2:
        curve_points = sample_bezier(control_points, NUM_SEGMENTS)
        for point in curve_points:
            x, y = normalized_to_pixel(point)
            image[y, x] = CURVE_COLOR

        for start, end in zip(control_points[:-1], control_points[1:]):
            draw_line_cpu(image, start, end, CONTROL_POLYGON_COLOR)

    for point in control_points:
        draw_disk_cpu(image, point, radius=6, color=CONTROL_POINT_COLOR)
    return image
def save_image(path, image):
    clipped = np.clip(image, 0.0, 1.0)
    rgb = (clipped * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_preview(path):
    image = build_preview_image(get_demo_points())
    save_image(path, image)


def get_demo_points():
    return [
        [0.12, 0.16],
        [0.28, 0.82],
        [0.68, 0.78],
        [0.86, 0.22],
    ]


def draw_scene(canvas, control_points):
    render_frame(control_points)
    canvas.set_image(pixels)
    canvas.lines(line_vertices, width=0.0025, color=CONTROL_POLYGON_COLOR)
    canvas.circles(
        gui_points,
        radius=CONTROL_POINT_RADIUS,
        color=CONTROL_POINT_COLOR,
        per_vertex_radius=gui_radii,
    )
def run():
    window = ti.ui.Window("CG Lab 3: Bezier Curve", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    control_points = []

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append([pos[0], pos[1]])
            elif event.key == "c":
                control_points.clear()
            elif event.key == ti.ui.ESCAPE:
                window.running = False

        draw_scene(canvas, control_points)
        window.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Bezier curve demo")
    parser.add_argument(
        "--save-preview",
        type=Path,
        help="save a preview image and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_preview:
        save_preview(args.save_preview)
        print(f"saved preview to {args.save_preview}")
        return
    run()


if __name__ == "__main__":
    main()
