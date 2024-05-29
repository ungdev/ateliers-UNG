import pygame
from keras.saving import load_model
import numpy as np
from keras.datasets import mnist

model = load_model("mnist.h5")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

window = pygame.display.set_mode((800, 800))
draw_surface = pygame.Surface((600, 600))
draw_surface.fill((255, 255, 255))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        if event.type == pygame.MOUSEMOTION and 100 <= event.pos[0] < 700 and 100 <= event.pos[1] < 700 and event.buttons[0] == 1:
            pygame.draw.circle(draw_surface, (0, 0, 0), (event.pos[0] - 100, event.pos[1] - 100), 30)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            draw_surface.fill((255, 255, 255))
    resized_surface = pygame.transform.smoothscale(draw_surface, (28, 28))
    resized_surface_data = 1 - (pygame.surfarray.array_red(resized_surface).transpose().reshape((784, )).astype('float32') / 255)
    print(np.argmax(model.predict(np.array([resized_surface_data]), verbose=0)))
    window.fill((0, 0, 0))
    window.blit(pygame.transform.scale(resized_surface, (600, 600)), (100, 100))
    pygame.display.flip()
