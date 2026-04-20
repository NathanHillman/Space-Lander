#####################################
# Nathaniel Hillman
# 04/16/2026
# MATH 371
# 
# Turning the volume on would be funny
#####################################

import pygame
import sys
import random
import math
import matplotlib.pyplot as plt
from time import sleep

# -----------------------------
# PRE_GAME QUESTIONS
# -----------------------------
difficulty = -1
step_type = ""

# Get the user's choice for how hard the game should be (How much fuel you get)
while difficulty == -1: 
    choice = input("What level of difficulty do you want (correponds to amount of fuel at start)?\n" \
    "0 - Easy\n" \
    "1 - Intermediate\n" \
    "2 - Hard\n" \
    "3 - Expert\n" \
    "4 - Impossible (Sometimes actually impossible)\n")

    if choice not in ["0", "1", "2", "3", "4"]:
        print("Please enter a valid difficulty level!\n")
    else:
        difficulty = int(choice)
        break

# Get the user's choice for how accurate the game's approximations will be
while step_type == "":
    choice = input("What method would you like to use for approximations?\n" \
    "rk4 -> Runge-Kutta 4th order\n" \
    "elr -> Euler's Method\n")

    if choice not in ["rk4", "elr"]:
        print("Please enter a valid option!\n")
    else:
        step_type = choice
        break


# -----------------------------
# INITIAL SETUP
# -----------------------------
pygame.init()

WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 24)
large_font = pygame.font.SysFont(None, 60)

lose_sound = pygame.mixer.Sound("files/lose_sound.mp3")
more_lose_sound = pygame.mixer.Sound("files/more_lose_sound.mp3")
win_sound = pygame.mixer.Sound("files/win_sound.mp3")
lose_sound.set_volume(1.0)
more_lose_sound.set_volume(1.0)
win_sound.set_volume(1.0)

pygame.display.set_caption("Moon Landing")
bg_image = pygame.image.load('files/starry_sky.png')

clock = pygame.time.Clock()

# -----------------------------
# PHYSICAL PARAMETERS
# -----------------------------
G = -1.6  # its the moon
DT = 0.05/2/2  # this is for the start only .  you need to adjust
M_DRY = 5000 # Mass of just the lander
m_starting_fuel = 3000.0 / (difficulty + 1) # Starting mass of total fuel
CONVERSION_FACTOR = .0007

NO_THRUST_RUN = False # Turn on to get no-thrust graphs
THRUST_FORCE_DOWN = 20000
THRUST_FORCE_SIDE = 2000 

# This way both are indexed to the size of the window
GROUND = ( HEIGHT / 12 )
ground_pixel = 11 * HEIGHT / 12

# -----------------------------
# SPRITE SIZES (IN METERS/PIXELS)
# -----------------------------
LANDER_H = 50
LANDER_W = 40 

THRUSTDOWN_H = 50
THRUSTDOWN_W = 40

# Because the side is then rotated, the width measurement 
#  becomes the height and vice versa.
THRUSTSIDE_H = 20
THRUSTSIDE_W = 25

# -----------------------------
# STATE VARIABLES
# -----------------------------
thrustingDown = False
thrustingRight = False
thrustingLeft = False


# -----------------------------
# SCREEN TRANSFORM
# -----------------------------
def to_screen(x, y):
    screen_y = int(ground_pixel - y)
    return x, screen_y

# -----------------------------
# LANDER CLASS (for easy compartmentalization)
# -----------------------------
class Lander:
    def __init__(self): 
        # Initialize parameters x, y, vx, vy, etc.       
        self.restart()

        # Other parameters not changed on restart
        self.height_of_fuel_bar = .5 # Takes up half the length of the screen
        self.width_of_fuel_bar = .05

        # lander images
        self.lander = pygame.image.load("files/lunarSprite.png").convert_alpha()
        self.lander = pygame.transform.scale(self.lander, (LANDER_W, LANDER_H))

        # Thrust images
        self.thrust = pygame.image.load("files/thrust.png").convert_alpha()
        self.thrustDownImage = pygame.transform.scale(self.thrust, (THRUSTDOWN_W,THRUSTDOWN_H))

        self.thrustSide = pygame.transform.scale(self.thrust, (THRUSTSIDE_W,THRUSTSIDE_H))
        self.thrustRightImage = pygame.transform.rotate(self.thrustSide, 90)
        self.thrustLeftImage = pygame.transform.rotate(self.thrustSide, -90)

    # -----------------------------
    # RESTART/INITIALIZE LANDER POSITION
    # -----------------------------
    def restart(self):
        self.x = random.randrange(0, WIDTH / 2) # Anywhere from the left edge to center

        # Going against the worksheet in order to have a larger y.
        #  Will now be from 1/3 to 2/3 height on the screen.
        #    > Translates to 250 to 450 meters on 600 x 600 screen.
        self.y = random.randrange(HEIGHT / 3, 2 * HEIGHT / 3) + GROUND

        self.vx = (random.random() * 4) - 2
        self.vy = 0

        self.ax = 0
        self.ay = G

        self.t = 0
        self.m_fuel = m_starting_fuel

        # The x value that the player will be steering towards. 49 differnt spots to choose from.
        self.x_target = (( WIDTH / 50 ) * random.randint(1, 49))

    # -----------------------------
    # THE DIFFERENTIAL EQUATIONS
    # -----------------------------
    def dynamics(self, t, Y, TL, TR, TB):
        x, y, vx, vy, m = Y

        # Total mass
        M = M_DRY + m

        dxdt = vx
        dydt = vy

        dvxdt = (TL - TR) / M
        dvydt = G + TB / M

        dMdt = -CONVERSION_FACTOR * (TL + TR + TB)

        return [dxdt, dydt, dvxdt, dvydt, dMdt]
    
    # -----------------------------
    # NUMERICAL METHOD (EULER)
    # -----------------------------
    def step_euler(self):
        Y = [self.x, self.y, self.vx, self.vy, self.m_fuel]
        dxdt, dydt, dvxdt, dvydt, dMdt = self.dynamics(self.t, Y, self.get_left_thrust(), self.get_right_thrust(), self.get_down_thrust())
        
        # position update
        self.x = self.x + DT * dxdt
        self.y = self.y + DT * dydt

        # velocity update
        self.vx = self.vx + DT * dvxdt
        self.vy = self.vy + DT * dvydt


        # mass update
        self.m_fuel += DT * dMdt
        if self.m_fuel < 0:
            self.m_fuel = 0

        # Time update
        self.t += DT


    # -----------------------------
    # NUMERICAL METHOD (Runge-Kutta)
    # -----------------------------    
    def step_runge_kutta(self):
        def f(t_local, Y_local):
            return self.dynamics(t_local, Y_local, self.get_left_thrust(), self.get_right_thrust(), self.get_down_thrust())
        
        # Starting values are stored in the class variables.
        Y = [self.x, self.y, self.vx, self.vy, self.m_fuel]

        # Implement the 4th order Runge-Kutta
        k1 = f(self.t, Y)

        k2 = f(self.t + DT/2, [
            Y[i] + DT*k1[i]/2 for i in range(len(Y))
        ])

        k3 = f(self.t + DT/2, [
            Y[i] + DT*k2[i]/2 for i in range(len(Y))
        ])

        k4 = f(self.t + DT, [
            Y[i] + DT*k3[i] for i in range(len(Y))
        ])

        self.x, self.y, self.vx, self.vy, self.m_fuel = (Y[i] + (DT/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(Y)))
        
        if self.m_fuel < 0:
            self.m_fuel = 0

        # Time update
        self.t += DT

    # -----------------------------
    # DRAW LANDER
    # -----------------------------
    def draw_lander(self):
        # lander
        screen_x, screen_y = to_screen(self.x, self.y)
        screen.blit(self.lander, (screen_x, screen_y))

        # Potentially draw thrusters
        if self.m_fuel > 0:
            if thrustingDown:
                # Change the x and y based on the lander's screen x & y
                thrust_x = screen_x + ((LANDER_W - THRUSTDOWN_W) / 2)
                thrust_y = screen_y + LANDER_H
                
                screen.blit(self.thrustDownImage, (thrust_x, thrust_y))

            if thrustingRight:
                # "Center" the thrust based on the lander's screen x & y
                thrust_x = screen_x + LANDER_W
                thrust_y = screen_y  + ((LANDER_H/2.3) - THRUSTSIDE_H/2)

                screen.blit(self.thrustRightImage, (thrust_x, thrust_y))

            if thrustingLeft:
                # "Center" the thrust based on the lander's screen x & y
                thrust_x = screen_x - THRUSTSIDE_H
                thrust_y = screen_y  + ((LANDER_H/2.4) - THRUSTSIDE_H/2)

                screen.blit(self.thrustLeftImage, (thrust_x, thrust_y))

    # -----------------------------
    # DRAW TARGET
    # -----------------------------
    def draw_target(self):
        # Drawing an X to mark the spot
        pygame.draw.line(screen, (255,255,255), 
                         (self.x_target - 10, ground_pixel - 10),
                         (self.x_target + 10, ground_pixel + 10),3)
        pygame.draw.line(screen, (255,255,255), 
                         (self.x_target - 10, ground_pixel + 10),
                         (self.x_target + 10, ground_pixel - 10),3)

    # -----------------------------
    # CHECK IF LANDER HIT GROUND
    # -----------------------------
    def on_ground(self):
        # Check to see if bottom of the rocket has touched down
        if self.y - LANDER_H <= 0:
            return True
        else:
            return False
        
    # -----------------------------
    # FUNCTIONS TO CALCULATE CRITICAL INFO
    # -----------------------------
    def get_total_m(self):
        return self.m_fuel + M_DRY
    
    def get_left_thrust(self):
        if thrustingLeft and self.m_fuel > 0:
            return THRUST_FORCE_SIDE
        else:
            return 0
        
    def get_right_thrust(self):
        if thrustingRight and self.m_fuel > 0:
            return THRUST_FORCE_SIDE
        else:
            return 0
        
    def get_down_thrust(self):
        if thrustingDown and self.m_fuel > 0:
            return THRUST_FORCE_DOWN
        else:
            return 0

    def get_mechanical_energy(self):
        m = self.get_total_m()
        return .5 * (m) * (self.vx**2 + self.vy**2) + m * (-G) * self.y
    
    # -----------------------------
    # FUNCTIONS TO CALCULATE WIN CONDITIONS
    # -----------------------------
    def get_magnitude(self):
        return math.sqrt(self.vx**2 + self.vy**2)
    
    def get_descent_angle(self):
        return abs(math.atan(self.vy/self.vx) * 180 / math.pi)
    
    def get_distance_to_target(self):
        return self.x_target - (self.x + LANDER_W / 2)
    
    # -----------------------------
    # FUNCTIONS TO CHECK WIN CONDITIONS (and the win itself)
    # -----------------------------
    # If it is good, return GREEN
    # Else return RED
    def check_magnitude(self):
        if self.get_magnitude() < 2:
            return (30,255,0)
        else:
            return (255,70,0)
        
    def check_descent_angle(self):
        if abs(self.get_descent_angle() - 90) < 2:
            return (30,255,0)
        else:
            return (255,70,0)
        
    def check_distance_to_target(self):
        if abs(self.get_distance_to_target()) < 1:
            return (30,255,0)
        else:
            return (255,70,0)
    
    # The win condition. Check to make sure ALL criterion are met
    def check_win(self):
        if self.get_magnitude() < 2 and abs(self.get_descent_angle() - 90) < 2 and abs(self.get_distance_to_target()) < 1:
            screen.fill((20, 20, 30))
            screen.blit(large_font.render("YOU WIN", True, (255,255,255)), 
                (WIDTH / 2 - 110, 100))
            win_sound.play()
            pygame.display.flip()
            
        else:
            screen.fill((0,0,0))
            screen.blit(large_font.render("CRASHED", True, (255,0,0)), 
                (WIDTH / 2 - 110, 100))
            
            lose_sound.play()
            pygame.display.flip()
            sleep(1)
            more_lose_sound.play()
        
        # Stay on screen for a little while
        sleep(8)
        
        
        
    
    # -----------------------------
    # FUNCTION TO DRAW CRITICAL INFO
    # -----------------------------
    def draw_lander_parameters(self):
        screen.blit(font.render(f"x = {self.x:.2f}", True, (255,255,255)), (5 * WIDTH / 6,10))
        screen.blit(font.render(f"y = {self.y:.2f}", True, (255,255,255)), (5 * WIDTH / 6,30))
        screen.blit(font.render(f"vx = {self.vx:.2f}", True, (255,255,255)), (5 * WIDTH / 6,50))
        screen.blit(font.render(f"vy = {self.vy:.2f}", True, (255,255,255)), (5 * WIDTH / 6,70))
        
    # -----------------------------
    # FUNCTION TO DRAW WIN CONDITION INFO
    # -----------------------------
    def draw_win_conditions(self):
        # The fuel guage. Change the color of the fuel guage as there is less and less fuel.
        percentRed = 2 * ((m_starting_fuel - self.m_fuel) / m_starting_fuel)
        percentGreen = ((self.m_fuel) / (m_starting_fuel / 2))

        # Drawing it on the right edge of the screen
        pygame.draw.rect(screen, (min(255 * percentRed, 255), 
                                  min(255 * percentGreen, 255), 0), 
                         pygame.Rect((29 * WIDTH / 30) - self.width_of_fuel_bar * WIDTH,
                            ((HEIGHT - self.height_of_fuel_bar * HEIGHT) / 2) + (1.0 - self.m_fuel / m_starting_fuel) * self.height_of_fuel_bar * HEIGHT, 
                            self.width_of_fuel_bar + WIDTH,
                            (self.height_of_fuel_bar * HEIGHT) * (self.m_fuel / m_starting_fuel)))
        
        # Print percentage of fuel to the left of the bar
        screen.blit(font.render(f"Fuel: {(self.m_fuel / m_starting_fuel) * 100:.2f}%", True, (255,255,255)), 
                    ((29 * WIDTH / 30) - self.width_of_fuel_bar * WIDTH - 50, (HEIGHT - self.height_of_fuel_bar * HEIGHT) / 2 - 27))
        
        # Print the 3 other win conditions to the screen
        screen.blit(font.render(f"Magnitude of Velocity: {self.get_magnitude():.2f} < 2", True, self.check_magnitude()), 
                    (10,10))
        screen.blit(font.render(f"Descent Angle: |{self.get_descent_angle():.2f}° - 90°| < 2°", True, self.check_descent_angle()), 
                    (10,30))
        screen.blit(font.render(f"Distance to Target: |{self.get_distance_to_target():.2f}| < 1", True, self.check_distance_to_target()), 
                    (10,50))


# -----------------------------
# START SCREEN LOOP
# -----------------------------
lander = Lander()
running = True
starting = False
while not starting and running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
                running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                starting = True
            
    screen.fill((20, 20, 30))

    screen.blit(font.render("*******************RULES*******************", True, (255,255,255)),
                (10, HEIGHT / 2 - 60))
    screen.blit(font.render("- Use the left, top, and right arrow keys to activate the thrusters.", True, (255,255,255)), 
                (10, HEIGHT / 2 - 40))
    screen.blit(font.render("- The fuel guage on the right side of the screen shows you how much", True, (255,255,255)), 
                (10, HEIGHT / 2 - 20))
    screen.blit(font.render("is left until the tank is empty!", True, (255,255,255)), 
                (10, HEIGHT / 2))
    screen.blit(font.render("- The top right shows the current positions and velocities", True, (255,255,255)), 
                (10, HEIGHT / 2 + 20))
    screen.blit(font.render("of the lander", True, (255,255,255)), 
                (10, HEIGHT / 2 + 40))
    screen.blit(font.render("- Land on the X with all 3 landing criterion (in top left) being green", True, (255,255,255)), 
                (10, HEIGHT / 2 + 60))
    screen.blit(font.render("to secure victory!", True, (255,255,255)), 
                (10, HEIGHT / 2 + 80))
    screen.blit(font.render("Press ENTER to start!", True, (255,255,255)), 
                (80, HEIGHT / 2 + 110))

    lander.draw_win_conditions()
    lander.draw_lander_parameters()

    pygame.display.flip()
    

# -----------------------------
# MAIN LOOP
# -----------------------------
# Lists for plots at the end
energies = []
times = []

# Now to start up the game
while running:   
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                lander.restart()

            if event.key == pygame.K_UP:
                thrustingDown = True

            if event.key == pygame.K_LEFT:
                thrustingRight = True

            if event.key == pygame.K_RIGHT:
                thrustingLeft = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                thrustingDown = False

            if event.key == pygame.K_LEFT:
                thrustingRight = False

            if event.key == pygame.K_RIGHT:
                thrustingLeft = False

    # -----------------------------
    # NUMERICAL STEP
    # -----------------------------
    if step_type == "elr":
        lander.step_euler()
    else:
        lander.step_runge_kutta()

    # -----------------------------
    # GET TOTAL ENERGY OF SYSTEM
    # -----------------------------
    if NO_THRUST_RUN:
        E = lander.get_mechanical_energy()
        energies.append(E)
        times.append(lander.t)

    # -----------------------------
    # STOP AT GROUND
    # -----------------------------
    if lander.on_ground():
        lander.check_win()
        break

    # -----------------------------
    # DRAW
    # -----------------------------
    # Draw the background image at (0, 0)
    screen.blit(bg_image, (0, 0))

    # ground
    pygame.draw.line(screen, (200, 200, 200),
                    (0, ground_pixel), (WIDTH, ground_pixel), 2)
    
    # draw the target
    lander.draw_target()

    # lander
    lander.draw_lander()

    # info
    lander.draw_lander_parameters()
    lander.draw_win_conditions()

    pygame.display.flip()

pygame.quit()

if NO_THRUST_RUN:
    # Plots for No Thrust Runs.
    plt.figure(figsize=(10, 6))
    plt.plot(times, energies, label=f"{step_type.upper()} Total Energy")
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"Total Energy vs Time ({step_type.upper()})")
    plt.grid(True)
    plt.legend()
    # plt.savefig(f"{step_type.upper()} No-Thrust Test", dpi=200)
    plt.show()

sys.exit()
