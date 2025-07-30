from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
import numpy as np

class DictGoalsWrappers(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({'observation':env.observation_space, 'goal':gym.spaces.Discrete(16)})
    def observation(self, observation):
        goal = self.env.env.current_goal_label_
        return {'observation': observation, 'goal': goal}
    
class MetaSimpleEnvReachOneGoal(MiniGridEnv):
    def __init__(
        self,
        seed_goal_number=42,
        size=10,
        agent_start_pos=None,
        agent_start_dir=None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        # so agent does not learn shortcut between goal indice and position
        rng_ = np.random.default_rng(seed_goal_number)
        self.goal_indices_ = rng_.choice(range(16), replace=False, size=16)
        positions = [(j, k) for j in range(2) for k in range(8)]
        self.goal_idx_to_goal_position = {i: positions[e] for e, i in enumerate(self.goal_indices_)}

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach goal"
    
    def _reward(self):
        return 1

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        # where_skip_wall = self._rand_int(1, height-1)
        for i in range(0, height):
            if i == 2:
                continue
            else:
                self.grid.set(5, i, Wall())        

        # Place a goal square in the bottom-right corner
        self.current_goal_label_ = self.goal_indices_[self._rand_int(0, 16)]
        x_shift, y_shift = self.goal_idx_to_goal_position[self.current_goal_label_]
        self.put_obj(Goal(), 7 + x_shift, 1 + y_shift)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


    
if __name__ == "__main__":
    env = MetaSimpleEnvReachOneGoal(render_mode="human")
           
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()