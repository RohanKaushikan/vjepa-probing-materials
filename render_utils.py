import numpy as np
import cv2

def render_frame(
    ball_x,
    ball_y,
    ball_radius,
    width,
    height,
    ball2_x=None,
    ball2_y=None,
    ball2_radius=None,
    ball2_color=(220, 230, 255),
    draw_floor=True,
):
    # constant dark grey backgroudn
    frame = np.full((height, width, 3), 40, dtype = np.uint8)

    # flip y-axis (pymunk y = 0 is at the bottom)
    screen_y = height - int(ball_y)
    screen_x = int(ball_x)

    # ball is a white filled circle
    cv2.circle(
        frame, (screen_x, screen_y), int(ball_radius), (255, 255, 255), -1
    )

    if ball2_x is not None and ball2_y is not None and ball2_radius is not None:
        sy2 = height - int(ball2_y)
        sx2 = int(ball2_x)
        cv2.circle(
            frame, (sx2, sy2), int(ball2_radius), ball2_color, -1
        )

    return frame


# Backward-compatible alias for earlier typo'd name.
render_fram = render_frame