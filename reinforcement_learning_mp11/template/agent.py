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

        # Don’t forget the edge case for updating your Q and N tables when t = 0. Both s and a will be None.
        if state is None or action is None:
            return
        # self.N[state][action] += 0.99        
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 

        # Don’t forget the edge case for updating your Q and N tables when t = 0. Both s and a will be None.
        if s is None or a is None:
            return
        
        alpha = self.C / (self.C + self.N[s][a])  # Learning rate

        max_q = np.max(self.Q[s_prime]) 

        self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * max_q - self.Q[s][a])      

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
    
        # When the agent “dies”, update Q and N for the action you just took that caused the death.
        if dead:
            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, -1, s_prime)
            self.reset()
            return utils.RIGHT # any arbitrary action can be chosen as the game will be reset before the action can be taken

        reward = -1 # -1 when the action causes the snake to die
        if points > self.points:
            reward = 1 # +1 when the action results in getting the food pellet
        elif points == self.points:
            reward = -0.1 # -0.1 otherwise (does not die nor get food)

        if self._train: # When testing, your agent no longer needs to update either table
            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)

        # Choosing the Optimal Action
        action_vals = []
        if self._train: # Exploratory Policy
            for action in self.actions:
                if self.N[s_prime][action] < self.Ne: 
                    action_vals.append(1)
                else:
                    action_vals.append(self.Q[s_prime][action])
        else: # When testing, choose the optimal action without the exploration function
            for action in self.actions:
                action_vals.append(self.Q[s_prime][action])
                
        # If there is a tie among actions, break it according to the priority order
        max_action_val = np.max(action_vals)
        max_action_indices = np.where(action_vals == max_action_val)[0]
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            if self.actions.index(action) in max_action_indices:
                priority_action = action
                break

        # Update states
        self.points = points
        self.s = s_prime
        self.a = priority_action

        return priority_action 

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
        if (snake_head_x == self.display_width - 2) or ((snake_head_x == rock_x - 1) and (snake_head_y == rock_y)):
            adjoining_wall_x = 2 # 2 (wall/rock on snake head right)
        if (snake_head_x == 1) or ((snake_head_x == rock_x + 2) and (snake_head_y == rock_y)):
            adjoining_wall_x = 1 # 1 (wall/rock on snake head left, or wall/rock on both snake head left and right)

        adjoining_wall_y = 0 # 0 (no adjoining wall/rock on y axis)
        if (snake_head_y == self.display_height - 2) or ((snake_head_y == rock_y - 1) and ((snake_head_x == rock_x) or (snake_head_x == rock_x + 1))):
            adjoining_wall_y = 2 # 2 (wall/rock on snake head bottom)
        if (snake_head_y == 1) or ((snake_head_y == rock_y + 1) and ((snake_head_x == rock_x) or (snake_head_x == rock_x + 1))):
            adjoining_wall_y = 1 # 1 (wall/rock on snake head top or wall/rock on both snake head top and bottom)

        if (snake_head_x < 1) or (snake_head_x > self.display_width - 2) or (snake_head_y < 1) or (snake_head_y > self.display_height - 2):
            adjoining_wall_x, adjoining_wall_y = 0, 0

        # Adjoining body detection
        adjoining_body_top = 0
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1 # 1 (adjoining top square has snake body)

        adjoining_body_bottom = 0
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1 # 1 (adjoining bottom square has snake body)

        adjoining_body_left = 0
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1 # 1 (adjoining left square has snake body)

        adjoining_body_right = 0
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1 # 1 (adjoining right square has snake body)

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)