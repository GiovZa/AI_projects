import heapq
# You do not need any other imports

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
    
    # Loop until the frontier is empty
    while frontier:
        # A priority queue follows FIFO:
        # Popping will pop the most recently added state.
        # The first state popped will be the starting_state.
        # Pop the state with the lowest cost (g + h)
        state = heapq.heappop(frontier)
        
        # Check if we've reached the goal
        if state.is_goal():
            return backtrack(visited_states, state)
        
        # For each current state popped from the frontier, its neighbors are explored. 
        # If a neighbor has not been visited or if it can be reached via a cheaper path than previously recorded, 
        # it's added to the frontier.
        for neighbor in state.get_neighbors():

            neighbor_distance = neighbor.dist_from_start
            
            # If the neighbor has not been visited or if we found a cheaper path to it, update the visited states.
            # we check for shorter paths to not add redundant or outdated paths.
            if neighbor not in visited_states or neighbor_distance < visited_states[neighbor][1]:
                # Update the visited states with the new visited state with parent and distance
                visited_states[neighbor] = (state, neighbor_distance)
                # Push the neighbor onto the priority queue.
                # This neighbor is now a state on the fronteir that will be popped and explored.
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

    # Start at the goal state
    current_state = goal_state
    
    # Trace back from the goal state to the starting state
    while current_state is not None: # The parent of the starting state is None, so stop there
        path.append(current_state)
        # Move to the parent state. visited_states[current_state][0] retrieves the parent_state of each tuple,
        # effectively backtracking through the states.
        current_state = visited_states[current_state][0]  
    
    # Reverse the path to go from start to goal so it is in the correct order
    path.reverse()
    # ------------------------------
    return path