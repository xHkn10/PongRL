import numpy as np
import random
from dataclasses import dataclass
from typing import Literal

SCREEN_WID = 70
SCREEN_HEI = 60


@dataclass
class Racket:
    pos: list[int]
    vel: tuple[int, int]
    side: Literal["Left", "Right"]
    score: int = 0
    len: int = 4  # half length actually
    curr_dir: int = 0  # 0 -> stationary, 1 -> down, -1 -> up


@dataclass
class Ball:
    pos: list[int]
    velocity: list[int]  # V_y, V_x


class PongGame:
    def __init__(self):
        # (y_l, x, y, v_y, v_x, y_r)
        self.racket_l = Racket([SCREEN_HEI // 2, 0], (1, 1), "Left")
        self.racket_r = Racket([SCREEN_HEI // 2, SCREEN_WID - 1], (1, 1), "Right")
        self.grid = np.zeros((SCREEN_HEI, SCREEN_WID))
        self.env_state = self.reset()

    def reset(self):
        self.racket_l = Racket([SCREEN_HEI // 2, 0], (1, 1), "Left", self.racket_l.score)
        self.racket_r = Racket([SCREEN_HEI // 2, SCREEN_WID - 1], (1, 1), "Right", self.racket_r.score)
        self.ball = Ball([SCREEN_HEI // 2, SCREEN_WID // 2], [random.choice((-1, 0, 1)), random.choice((-1, 1))])
        self._update_grid()
        return self.racket_l.pos[0], *self.ball.pos, *self.ball.velocity, self.racket_r.pos[0]

    def mirror_state(self, state):
        return state[-1], state[1], SCREEN_WID - state[2], state[3], -state[4], state[0]

    def step(self, action, wind=True, left_bot=False, right_bot=False):
        # 0 up 1 down 2 nothing

        if left_bot:
            self.racket_l.pos = [min(SCREEN_HEI - self.racket_l.len, max(self.racket_l.len, self.ball.pos[0])), 0]
        if right_bot:
            self.racket_r.pos = [min(SCREEN_HEI - self.racket_l.len, max(self.racket_l.len, self.ball.pos[0])),
                                 SCREEN_WID - 1]

        self._take_action(action)
        self._move_ball(wind)

        reward = .0
        is_done = False

        if self._is_collision():
            reward = 1.
        else:
            whose_point = self._check_point_scored()
            if whose_point == 1:
                reward = 6.
            elif whose_point == -1:
                reward = -6.
            if whose_point != 0:
                is_done = True

        self._update_grid()
        self.env_state = self.racket_l.pos[0], *self.ball.pos, *self.ball.velocity, self.racket_r.pos[0]
        return self.env_state, reward, is_done

    def _move_ball(self, wind):
        self.ball.pos[0] += self.ball.velocity[0]
        self.ball.pos[1] += self.ball.velocity[1]
        if wind and random.random() < .05 and 5 < self.ball.pos[1] < SCREEN_WID - 5:
            self.ball.pos[0] += random.choice([-1, 1])
            self.ball.pos[1] += random.choice([-1, 1])
        self.ball.pos[0] = int(np.clip(self.ball.pos[0], 0, SCREEN_HEI - 1))
        self.ball.pos[1] = int(np.clip(self.ball.pos[1], 0, SCREEN_WID - 1))
        return self.ball.pos

    def _take_action(self, action, racket="Left"):
        racket = self.racket_l if racket == "Left" else self.racket_r
        if action == 0:
            racket.pos[0] = max(racket.pos[0] - racket.vel[0], racket.len)
            racket.curr_dir = -1
        elif action == 1:
            racket.pos[0] = min(racket.pos[0] + racket.vel[0], SCREEN_HEI - racket.len)
            racket.curr_dir = 1
        elif action == 2:
            racket.curr_dir = 0
        else:
            raise Exception

    def _is_collision(self, agent="Left"):
        def check_racket_coll(racket):
            bool1 = racket.pos[0] - racket.len <= self.ball.pos[0] < racket.pos[0] + racket.len
            if not bool1: return False
            off_by_one = 1 if racket is self.racket_l else -1
            bool2 = racket.pos[1] + off_by_one == self.ball.pos[1]
            return bool2

        res = None

        if check_racket_coll(self.racket_l):
            res = agent == "Left"
        elif check_racket_coll(self.racket_r):
            res = agent == "Right"
        if not 0 < self.ball.pos[0] < SCREEN_HEI - 1:
            self.ball.velocity[0] *= -1
        if res is not None:
            self.ball.velocity[1] *= -1
            coll_racket = self.racket_l if res == "Left" else self.racket_r
            if random.random() < .8:
                self.ball.velocity[0] = coll_racket.curr_dir
            else:
                self.ball.velocity[0] = random.choice((-1, 0, 1))
        return res

    def _check_point_scored(self):
        """
        Return -1 if right's point, 1 if left's point, else 0
        """
        if self.ball.pos[1] <= 0:
            self.reset()
            self.racket_r.score += 1
            return -1
        elif self.ball.pos[1] >= SCREEN_WID - 1:
            self.reset()
            self.racket_l.score += 1
            return 1
        return 0

    def _update_grid(self):
        self.grid.fill(0.)
        self.grid[*self.ball.pos] = 1.
        r_l_slice = slice(self.racket_l.pos[0] - self.racket_l.len, self.racket_l.pos[0] + self.racket_l.len)
        r_r_slice = slice(self.racket_r.pos[0] - self.racket_r.len, self.racket_r.pos[0] + self.racket_r.len)
        self.grid[r_l_slice, 0] = 1.
        self.grid[r_r_slice, -1] = 1.


if __name__ == '__main__':
    game = PongGame()
    import matplotlib.pyplot as plt

    plt.ion()  # Turn on interactive mode

    while True:
        plt.imshow(game.grid)
        plt.draw()  # Draw the updated plot
        plt.text(15, .5, f"{game.racket_l.score, game.racket_r.score}", ha="center", va="center",
                 fontsize=20, weight="bold")
        plt.pause(0.001)  # Pause briefly
        plt.clf()  # Clear the figure for the next frame

        game.step(2)
