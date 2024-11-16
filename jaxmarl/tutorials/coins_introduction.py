"""
Short introduction to running the Overcooked environment and visualising it using random actions.
"""

import jax 
from jaxmarl import make
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
from jaxmarl.viz.visualizer import Visualizer, SMAXVisualizer
import time

# Parameters + random keys
max_steps = 300
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
layout = overcooked_layouts["cramped_room"]

# Or make your own!
# custom_layout_grid = """
# WWOWW
# WA  W
# B P X
# W  AW
# WWOWW
# """
# layout = layout_grid_to_dict(custom_layout_grid)

# Instantiate environment
# env = make('overcooked', layout=layout, max_steps=max_steps)

env = make('coin_game')

obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
print('example action dict', actions)

state_seq = []
pics = []
for _ in range(max_steps):
    state_seq.append(state)
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)
    key_a = jax.random.split(key_a, env.num_agents)

    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)

    img = env.render(state)
    pics.append(img)
    print(rewards)

# viz = Visualizer(env, state_seq)

# viz.animate(view=False, save_fname="coins.gif")
# viz = OvercookedVisualizer()
# # Render to screen
# for s in state_seq:
#     viz.render(env.agent_view_size, s, highlight=False)
#     time.sleep(0.25)

# # Or save an animation
# viz.animate(state_seq, agent_view_size=5, filename='animation.gif')
# viz.animate(view=False, save_fname="output.gif")

pics[0].save(
    "test1.gif",
    format="gif",
    save_all=True,
    append_images=pics[1:],
    duration=300,
    loop=0,
)