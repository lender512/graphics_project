import cv2
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
GRAVITY = 9.8
FRICTION = 0.99
MAX_BALLS = 10

# Ball class
class Ball:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 10

    def update(self, dt):
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt + 0.5 * GRAVITY * dt**2

        # Update velocity
        self.vy += GRAVITY * dt

        # Apply friction
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Bounce off walls
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx *= -1
        if self.y + self.radius > HEIGHT:
            self.y = HEIGHT - self.radius
            self.vy *= -0.8

# Initialize variables
balls = []
dragging = False
start_pos = None
mouse_pos = None
line_color = None
# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global dragging, start_pos, mouse_pos, line_color
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_pos = (x, y)
        line_color = tuple(np.random.randint(0, 255, 3).tolist())
        

    elif event == cv2.EVENT_LBUTTONUP:
        if dragging and start_pos:
            end_pos = (x, y)
            vx = -(end_pos[0] - start_pos[0]) / 5
            vy = -(end_pos[1] - start_pos[1]) / 5
            ball = Ball(start_pos[0], start_pos[1], vx, vy)
            ball.color = line_color
            balls.append(ball)
        dragging = False
        start_pos = None
    
    if dragging and start_pos:
        mouse_pos = (x, y)
        

# Create window and set mouse callback
cv2.namedWindow("Ball Simulation")
cv2.setMouseCallback("Ball Simulation", mouse_callback)

# Main loop
while True:
    # Create a blank image
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Draw balls
    for ball in balls:
        cv2.circle(img, (int(ball.x), int(ball.y)), ball.radius, ball.color, -1)

    # Draw drag line
    if dragging and start_pos:
        cv2.line(img, start_pos, mouse_pos, line_color, 2)

    # Update ball positions
    for ball in balls:
        ball.update(0.1)

    # Remove balls that are out of bounds
    balls = [ball for ball in balls if 0 <= ball.x <= WIDTH and ball.y <= HEIGHT]

    # Display the image
    cv2.imshow("Ball Simulation", img)

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()