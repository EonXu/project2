from gym.envs.registration import register

register(
    id='wolfpack-v0',
    entry_point='envs.Wolfpack.wolfpack_penalty_open:WolfpackPenaltyOpen',
    kwargs={
        'grid_height': 20,
        'grid_width': 20,
        'num_agents': 3,
        'seed': 1,
        'close_penalty': 0.1,
        'max_player_num': 5
    }
)
