import pygame
import torch
from ppo import PolicyValueNet
from pong_env import PongGame, SCREEN_WID, SCREEN_HEI
from DQN_PONG_features.model import DQN

pygame.init()

multiplier = 12
WIDTH = multiplier * SCREEN_WID
HEIGHT = multiplier * SCREEN_HEI

black = 0, 0, 0
white = 255, 255, 255

font = pygame.font.Font(None, 60)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock()


def draw(game):
    rect_l = 0, (game.racket_l.pos[0] - game.racket_l.len) * multiplier, multiplier, multiplier * game.racket_l.len * 2
    rect_r = (WIDTH - multiplier,
              (game.racket_r.pos[0] - game.racket_r.len) * multiplier, multiplier, multiplier * game.racket_r.len * 2)
    pygame.draw.rect(screen, white, rect_l)
    pygame.draw.rect(screen, white, rect_r)
    pygame.draw.circle(screen, white, (game.ball.pos[1] * multiplier, game.ball.pos[0] * multiplier), multiplier * .8)

    text = font.render(f"{game.racket_l.score}-{game.racket_r.score}", True, white)
    text_rec = text.get_rect(center=(WIDTH / 2, 20))
    screen.blit(text, text_rec)


def main(game, model_l, model_r):
    running = True
    state = game.reset()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action_l = model_l.act(state[:-1], temperature=.1)
        action_r = model_r.act(game.mirror_state(state), temperature=.1)
        key_pressed = pygame.key.get_pressed()
        if key_pressed[pygame.K_UP]:
            game._take_action(0, "Right")
        elif key_pressed[pygame.K_DOWN]:
            game._take_action(1, "Right")
        else:
            game._take_action(2, "Right")
        game._take_action(action_r, "Right")
        state, _, _ = game.step(action_l, wind=False, right_bot=False)

        draw(game)

        pygame.display.flip()
        clock.tick(35)
        screen.fill(black)


if __name__ == '__main__':
    game = PongGame()
    model_l = DQN(3)
    model_l.load_state_dict(torch.load("../DQN_PONG_features/models/model_5ft8000000.pth"))
    model_r = PolicyValueNet(6, 64, 3)
    model_r.load_state_dict(torch.load("models/model3050_dest.pth"))
    model_l.eval()
    model_r.eval()
    main(game, model_l, model_r)
