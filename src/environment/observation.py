def update_movement_flags(player_in_turn, movement_flags, movement_flags_memory, k):

    movement_flags_memory[player_in_turn] = k

    for i in range(len(movement_flags)):

        if movement_flags_memory[i] > 0:
            movement_flags[i] = 1
            movement_flags_memory[i] = movement_flags_memory[i] - 1

        else:
            movement_flags[i] = 0
            movement_flags_memory[i] = 0

    return movement_flags, movement_flags_memory