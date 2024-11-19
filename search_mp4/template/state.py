from abc import ABC, abstractmethod
from itertools import count, product
import numpy as np

from utils import compute_mst_cost

# NOTE: using this global index (for tiebreaking) means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()

# Manhattan distance between two (x,y) points
# Manhattan distance between two points (a=(a1,a2), b=(b1,b2))
def manhattan(a, b):
    manhattan_distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return manhattan_distance

class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # Return True if self is less than other
    # This method allows the heap to sort States according to f = g + h value
    def __lt__(self, other):    
        # You should return True if the current state has a lower g + h value than "other"
        # If they have the same value then you should use tiebreak_idx to decide which is smaller

        # h = Heuristic Estimate to Goal
        # g = Distance from Start
        # f = Total Estimated Cost

        # f = g + h
        self_f = self.dist_from_start + self.h
        other_f = other.dist_from_start + other.h
        
        is_self_cheaper = self_f < other_f
        is_other_cheaper = self_f > other_f
        lowest_index_wins_tie = self.tiebreak_idx < other.tiebreak_idx

        if is_self_cheaper:
            return True
        elif is_other_cheaper:
            return False
        else:
            return lowest_index_wins_tie

    # __hash__ method allow us to keep track of which 
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass
    
class SingleGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (x, y)
    # goal: a length 2 tuple indicating the goal location, e.g., (x, y)
    # maze_neighbors: function for finding neighbors on the grid (deals with checking collision with walls...)
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        self.maze_neighbors = maze_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # This is basically just a wrapper for self.maze_neighbors
    def get_neighbors(self):
        nbr_states = []
        # neighboring_locs is a tuple of tuples of neighboring locations, e.g., ((x1, y1), (x2, y2), ...)
        # feel free to look into maze.py for more details
        neighboring_locs = self.maze_neighbors(*self.state)
        # TODO(III): fill this in
        # The distance from the start to a neighbor is always 1 more than the distance to the current state
        # -------------------------------
        for nbr in neighboring_locs:
            new_state = SingleGoalGridState(
                state=nbr, 
                goal=self.goal, 
                dist_from_start=self.dist_from_start + 1, 
                use_heuristic=self.use_heuristic, 
                maze_neighbors=self.maze_neighbors
            )
            nbr_states.append(new_state)
        # -------------------------------
        return nbr_states

    # TODO(III): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    # Your heuristic should be the manhattan distance between the state and the goal
    
    def is_goal(self):
        return self.state == self.goal
    
    def compute_heuristic(self):
        return manhattan(self.state, self.goal)
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        return self.state == other.state
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return str(self.state)

class MultiGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (x, y)
    # goal: a tuple of length 2 tuples of locations in the grid that have not yet been reached
    #       e.g., ((x1, y1), (x2, y2), ...)
    # maze_neighbors: function for finding neighbors on the grid (deals with checking collision with walls...)
    # mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, mst_cache):
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # We get the list of neighbors from maze_neighbors
    # Then we need to check if we've reached one of the goals, and if so remove it
    def get_neighbors(self):
        nbr_states = []
        # neighboring_locs is a tuple of tuples of neighboring locations, e.g., ((x1, y1), (x2, y2), ...)
        # feel free to look into maze.py for more details
        neighboring_locs = self.maze_neighbors(*self.state)
        # TODO(IV): fill this in
        # -------------------------------

        for potential_nbr in neighboring_locs:
            remaining_goals = []

            for goal in self.goal:
                if goal == potential_nbr: # do not add goal if it is a neighbor
                    continue
                remaining_goals.append(goal)
                    
            new_goals = tuple(remaining_goals)
            
            new_state = MultiGoalGridState(
                state=potential_nbr, 
                goal=new_goals, 
                dist_from_start=self.dist_from_start + 1, 
                use_heuristic=self.use_heuristic, 
                maze_neighbors=self.maze_neighbors,
                mst_cache=self.mst_cache
            )
            nbr_states.append(new_state)

        # -------------------------------
        return nbr_states

    # TODO(IV): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    # Your heuristic should be the cost of the minimum spanning tree of the remaining goals 
    #   plus the manhattan distance to the closest goal
    #   (you should use the mst_cache to store the MST values)
    # Think very carefully about your eq and hash methods, is it enough to just hash the state?
    
    def is_goal(self):
        return len(self.goal) == 0 # distance to goal is 0
    
    def compute_heuristic(self):
        # Check cache for MST cost; compute and store if not cached
        mst_cost = self.mst_cache.get(self.goal)
        if mst_cost is None:
            mst_cost = compute_mst_cost(self.goal, manhattan)
            self.mst_cache[self.goal] = mst_cost
        
        # Find the closest goal if any, otherwise set to 0
        dist_to_closest_goal = min(manhattan(self.state, goal) for goal in self.goal) if self.goal else 0
        
        return mst_cost + dist_to_closest_goal

    
    def __hash__(self): # sorted tuple to not get duplicate states
        return hash((self.state, tuple(sorted(self.goal))))
    
    def __eq__(self, other): # use set to get unique and unordered grouping
        return self.state == other.state and set(self.goal) == set(other.goal)
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    
class MultiAgentGridState(AbstractState):
    # state: a tuple of agent locations
    # goal: a tuple of goal locations for each agent
    # maze_neighbors: function for finding neighbors on the grid
    #   NOTE: it deals with checking collision with walls... but not with other agents
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, h_type="admissible"):
        self.maze_neighbors = maze_neighbors
        self.h_type = h_type
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # get_neighbors helper function to reduce nested statements
    def agents_cross_paths(self, nbr_locs):
        # Check if any two agents cross paths by swapping locations
        for a in range(len(self.state)):
            for b in range(a + 1, len(self.state)):
                if self.state[a] == nbr_locs[b] and self.state[b] == nbr_locs[a]:
                    return True  # Agents cross paths
        return False
        
    # We get the list of neighbors for each agent from maze_neighbors
    # Then we need to check inter agent collision and inter agent edge collision (crossing paths)
    def get_neighbors(self):
        nbr_states = []
        neighboring_locs = [self.maze_neighbors(*s) for s in self.state]
        for nbr_locs in product(*neighboring_locs):
            # TODO(V): fill this in
            # You will need to check whether two agents collide or cross paths
            #   - Agents collide if they move to the same location 
            #       - i.e., if there are any duplicate locations in nbr_locs
            #   - Agents cross paths if they swap locations
            #       - i.e., if there is some agent whose current location (in self.state) 
            #       is the same as the next location of another agent (in nbr_locs) *and vice versa*
            # Before writing code you might want to understand what the above lines of code do...
            # -------------------------------
            if len(set(nbr_locs)) != len(nbr_locs):
                continue  # Skip if agents collide

            if self.agents_cross_paths(nbr_locs):
                continue  # Skip if agents cross paths

            # If no collisions or crossing paths are detected, a new state is created
            new_state = MultiAgentGridState(
                state=nbr_locs,
                goal=self.goal,
                dist_from_start=self.dist_from_start + 1,
                use_heuristic=self.use_heuristic,
                maze_neighbors=self.maze_neighbors,
                h_type=self.h_type
            )
            nbr_states.append(new_state)
            # -------------------------------            
        return nbr_states
    
    def compute_heuristic(self):
        if self.h_type == "admissible":
            return self.compute_heuristic_admissible()
        elif self.h_type == "inadmissible":
            return self.compute_heuristic_inadmissible()
        else:
            raise ValueError("Invalid heuristic type")

    # TODO(V): fill in the compute_heuristic_admissible and compute_heuristic_inadmissible methods
    #   as well as the is_goal, __hash__, and __eq__ methods
    # As implied, in compute_heuristic_admissible you should implement an admissible heuristic
    #   and in compute_heuristic_inadmissible you should implement an inadmissible heuristic 
    #   that explores fewer states but may find a suboptimal path
    # Your heuristics should be at least as good as ours on the autograder 
    #   (with respect to number of states explored and path length)

    def compute_heuristic_admissible(self):
        """
        Admissible heuristic: Ensures that the estimated cost to reach the goal never exceeds
        the actual cost. This allows the search to explore more paths without underestimating
        the true cost, guaranteeing an optimal solution in search algorithms like A*.
        """
        # Compute the maximum Manhattan distance between any agent and its goal
        # This ensures the heuristic is optimistic and admissible
        total = max(manhattan(state, goal) for state, goal in zip(self.state, self.goal))
        
        # Add the Minimum Spanning Tree (MST) cost for all remaining goals, 
        # estimating the minimum extra cost to visit all goals
        total += compute_mst_cost(self.state, manhattan)
        
        return total
    
    def compute_heuristic_inadmissible(self):
        """
        Non-admissible heuristic: Encourages focus on certain paths by slightly overestimating
        the cost, which can help reduce exploration in open mazes with many similar paths.
        """
        # Overestimate the total cost by summing Manhattan distances for all agents
        return sum(manhattan(state, goal) for state, goal in zip(self.state, self.goal))

    
    def is_goal(self):
        for i in range(len(self.state)):
            if self.state[i] != self.goal[i]:
                return False
        return True # Only true if all agents are at their respective goals

    def __eq__(self, other):
        return self.state == other.state and self.goal == other.goal

    def __hash__(self):
        return hash((self.state, self.goal))
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)