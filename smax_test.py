from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.viz.visualizer import SMAXVisualizer

scenario = map_name_to_scenario("3m")
env = make(
    "HeuristicEnemySMAX",
    enemy_shoots=True,
    scenario=scenario,
    num_agents_per_team=3,
    use_self_play_reward=False,
    walls_cause_death=True,
    see_enemy_actions=False,
)

# state_seq is a list of (key_s, state, actions) tuples
# where key_s is the RNG key passed into the step function,
# state is the jax env state and actions is the actions passed
# into the step function.
# viz = SMAXVisualizer(env, state_seq)

# viz.animate(view=False, save_fname="output.gif")