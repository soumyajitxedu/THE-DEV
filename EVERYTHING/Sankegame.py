from tkinter import Tk, Canvas, Label
import random

GAME_WIDTH = 700
GAME_HEIGHT = 700
SPEED = 120
SPACE_SIZE = 25
BODY_PARTS = 3
SNAKE_COLOR = "#BD4A4A"
FOOD_COLOR = "#1F6F7E"
BACKGROUND = "#DB1561"


class Snake:
    def __init__(self):
        self.body_size = BODY_PARTS
        self.coordinates = []
        self.squares = []

        for i in range(0, BODY_PARTS):
            self.coordinates.append([0, 0])

    def reset(self):
        self.body_size = BODY_PARTS
        self.coordinates = [[0, 0] for _ in range(BODY_PARTS)]
        for sq in self.squares:
            canvas.delete(sq)
        self.squares = []


class Food:
    def __init__(self):
        self.coordinates = [0, 0]
        self.square = None


def next_turn(snake_obj, food_obj):
    global direction, score

    x, y = snake_obj.coordinates[0]
    if direction == 'up':
        y -= SPACE_SIZE
    elif direction == 'down':
        y += SPACE_SIZE
    elif direction == 'left':
        x -= SPACE_SIZE
    elif direction == 'right':
        x += SPACE_SIZE

    new_head = [x, y]
    snake_obj.coordinates.insert(0, new_head)

    square = canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR, tags="snake")
    snake_obj.squares.insert(0, square)

    if x == food_obj.coordinates[0] and y == food_obj.coordinates[1]:
        score += 1
        label.config(text=f"Score: {score}")
        canvas.delete(food_obj.square)
        food_obj = place_food()
    else:
        # remove last
        del snake_obj.coordinates[-1]
        canvas.delete(snake_obj.squares[-1])
        del snake_obj.squares[-1]

    if check_collisions(snake_obj):
        game_over()
    else:
        window.after(SPEED, next_turn, snake_obj, food_obj)


def change_direction(new_direction):
    global direction
    all_directions = ({'left', 'right', 'up', 'down'})
    opposites = ({'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'})
    if new_direction in all_directions:
        if opposites.get(new_direction) != direction:
            direction = new_direction


def check_collisions(snake_obj):
    x, y = snake_obj.coordinates[0]

    if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
        return True

    # Check self collision
    for body_part in snake_obj.coordinates[1:]:
        if x == body_part[0] and y == body_part[1]:
            return True

    return False


def game_over():
    canvas.delete("all")
    canvas.create_text(GAME_WIDTH / 2, GAME_HEIGHT / 2, font=('consolas', 70), text="GAME OVER", fill="red")


def place_food():
    x = random.randint(0, (GAME_WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
    y = random.randint(0, (GAME_HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
    food = Food()
    food.coordinates = [x, y]
    food.square = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=FOOD_COLOR, tags="food")
    return food


window = Tk()
window.title("Snake Game")
window.resizable(False, False)

score = 0
direction = 'down'

label = Label(window, text=f"Score: {score}", font=('consolas', 20))
label.pack()

canvas = Canvas(window, bg=BACKGROUND, height=GAME_HEIGHT, width=GAME_WIDTH)
canvas.pack()

window.update()
window_width = window.winfo_width()
window_height = window.winfo_height()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

snake = Snake()
food = place_food()

for x, y in snake.coordinates:
    square = canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR)
    snake.squares.append(square)

window.bind('<Left>', lambda event: change_direction('left'))
window.bind('<Right>', lambda event: change_direction('right'))
window.bind('<Up>', lambda event: change_direction('up'))
window.bind('<Down>', lambda event: change_direction('down'))

next_turn(snake, food)

window.mainloop()

