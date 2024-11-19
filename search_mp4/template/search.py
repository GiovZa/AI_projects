import heapq

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------

   # Loop through starting_state and neighbors until goal is found
    while frontier:
        # States are popped in order of lowest cost / highest priority
        state = heapq.heappop(frontier)
        
        # Once goal is found, reconstruct path using backtrack
        if state.is_goal():
            return backtrack(visited_states, state)
        
        # Get all adjacent states to current state
        for neighbor in state.get_neighbors():

            # Retrieve distance for each neighbor
            neighbor_distance = neighbor.dist_from_start

            # Push neighbor to frontier if neighbor has not been visited or can be reached with a shorter path
            if neighbor not in visited_states or neighbor_distance < visited_states[neighbor][1]:
                visited_states[neighbor] = (state, neighbor_distance)
                heapq.heappush(frontier, neighbor) 

    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    # Your code here ---------------

    # Must work backwards
    current_state = goal_state
    
    # Loop until we get starting_state's None parent
    while current_state is not None:
        path.append(current_state)
        # Get current_state's parent since each parent contains lowest cost from starting_state
        current_state = visited_states[current_state][0]  
    
    # Reverse since we want start -> goal
    path.reverse()
    
    # ------------------------------
    return path