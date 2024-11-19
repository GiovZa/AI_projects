import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 

        # if state is None or action is None:
        #     return
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        if s is None or a is None:
            return  # No updates on the first step
        
        n = self.N[s][a]
        alpha = self.C / (self.C + n)  # Learning rate

        max_q = np.max(self.Q[s_prime]) if s_prime is not None else 0  # Handle terminal state
        self.Q[s][a] += alpha * (r + self.gamma * max_q - self.Q[s][a])        

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 

        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        # Food direction
        food_dir_x = 0 # 0 (same coords on x axis)
        if food_x < snake_head_x:
            food_dir_x = 1  # 1 (food on snake head left)
        elif food_x > snake_head_x:
            food_dir_x = 2 # 2 (food on snake head right)

        food_dir_y = 0 # 0 (same coords on y axis)
        if food_y < snake_head_y:
            food_dir_y = 1 # 1 (food on snake head top)
        elif food_y > snake_head_y:
            food_dir_y = 2 # 2 (food on snake head bottom)

        # Adjoining wall/rock detection
        adjoining_wall_x = 0 # 0 (no adjoining wall/rock on x axis)
        if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y):
            adjoining_wall_x = 1 # 1 (wall/rock on snake head left, or wall/rock on both snake head left and right)
        elif snake_head_x == self.display_width or (snake_head_x == rock_x - 1 and snake_head_y == rock_y):
            adjoining_wall_x = 2 # 2 (wall/rock on snake head right)

        adjoining_wall_y = 0 # 0 (no adjoining wall/rock on y axis)
        if snake_head_y == 1 or (snake_head_y == rock_y + 1 and (snake_head_x == rock_x or snake_head_x == rock_x + 2)):
            adjoining_wall_y = 1 # 1 (wall/rock on snake head top or wall/rock on both snake head top and bottom)
        elif snake_head_y == self.display_height or (snake_head_y == rock_y - 1 and (snake_head_x == rock_x or snake_head_x == rock_x + 2)):
            adjoining_wall_y = 2 # 2 (wall/rock on snake head bottom)

        # Adjoining body detection
        adjoining_body_top = 0
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1

        adjoining_body_bottom = 0
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1

        adjoining_body_left = 0
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1

        adjoining_body_right = 0
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)