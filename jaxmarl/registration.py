from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    SMAX,
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
    SwitchRiddle,
    Ant,
    Humanoid,
    Hopper,
    Walker2d,
    HalfCheetah,
    InTheGrid,
    InTheMatrix,
    Hanabi,
    Overcooked,
    CoinGame,
    JaxNav,
    harvest_common_open
)



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # 5. InTheGrid
    elif env_id == "storm":
        env = InTheGrid(**env_kwargs)
    # 5. InTheGrid
    # elif env_id == "storm_2p":
    #     env = InTheGrid_2p(**env_kwargs)
    elif env_id == "storm_np":
        env = InTheMatrix(**env_kwargs)

    elif env_id == "harvest_common_open":
        env = InTheMatrix(**env_kwargs)
    

    # 7. Overcooked
    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)

    # 8. Coin Game
    elif env_id == "coin_game":
        env = CoinGame(**env_kwargs)

    elif env_id == "SMAX":
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**env_kwargs)

        


    return env

registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    "storm",
    "storm_2p",
    "storm_np",
    "hanabi",
    "overcooked",
    "coin_game",
    "jaxnav",
    "harvest_common_open"
]
