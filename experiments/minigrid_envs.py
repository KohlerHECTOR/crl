from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Key, Door, Box
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



class SimpleEnvReachOneGoal(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=None,
        agent_start_dir=None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall with opening at i==2
        for i in range(0, height):
            if i == 2:
                continue
            else:
                self.grid.set(5, i, Wall())        

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


# class ParametricDoorKeyEnv(MiniGridEnv):
#     """
#     Environment with a parametric door key task.
#     Parameter: key_position - tuple (x, y) where to place the key (must be on agent's side)
#     """
#     def __init__(
#         self,
#         key_position=None,
#         size=10,
#         agent_start_pos=None,
#         agent_start_dir=None,
#         max_steps: int | None = None,
#         **kwargs,
#     ):
#         self.key_position = key_position
#         self.agent_start_pos = agent_start_pos
#         self.agent_start_dir = agent_start_dir

#         mission_space = MissionSpace(mission_func=self._gen_mission)

#         if max_steps is None:
#             max_steps = 4 * size**2

#         super().__init__(
#             mission_space=mission_space,
#             grid_size=size,
#             see_through_walls=True,
#             max_steps=max_steps,
#             **kwargs,
#         )

#     @staticmethod
#     def _gen_mission():
#         return "pick up the key and open the door to reach the goal"

#     def _gen_grid(self, width, height):
#         # Create an empty grid
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.wall_rect(0, 0, width, height)

#         # Generate vertical separation wall with opening at i==2
#         for i in range(0, height):
#             if i == 2:
#                 continue
#             else:
#                 self.grid.set(5, i, Wall())

#         # Randomize door color
#         door_colors = ["red", "green", "blue", "purple", "yellow", "grey"]
#         door_color = self._rand_elem(door_colors)

#         # Place a door at x==5 and i==2 (row 2)
#         door_x = 5
#         door_y = 2
#         self.put_obj(Door(door_color, is_locked=True), door_x, door_y)

#         # Place the agent first to determine which side the key should be on
#         if self.agent_start_pos is not None:
#             self.agent_pos = self.agent_start_pos
#             self.agent_dir = self.agent_start_dir
#         else:
#             self.place_agent()

#         # Determine key position - must be on agent's side (left of wall at x=5)
#         if self.key_position is None:
#             # Random position on agent's side
#             if self.agent_pos[0] < 5:  # Agent is on left side
#                 key_x = self._rand_int(1, 4)
#                 key_y = self._rand_int(1, height - 2)
#             else:  # Agent is on right side
#                 key_x = self._rand_int(6, width - 2)
#                 key_y = self._rand_int(1, height - 2)
#         else:
#             key_x, key_y = self.key_position

#         # Place the key with matching color
#         self.put_obj(Key(door_color), key_x, key_y)

#         # Randomize goal position on the other side of the wall
#         if self.agent_pos[0] < 5:  # Agent is on left side
#             goal_x = self._rand_int(6, width - 2)
#             goal_y = self._rand_int(1, height - 2)
#         else:  # Agent is on right side
#             goal_x = self._rand_int(1, 4)
#             goal_y = self._rand_int(1, height - 2)

#         self.put_obj(Goal(), goal_x, goal_y)

#         self.mission = "pick up the key and open the door to reach the goal"


# class ParametricGoalReachEnv(MiniGridEnv):
#     """
#     Environment with a parametric goal reach task.
#     Parameter: goal_position - tuple (x, y) where to place the goal (None for random)
#     """
#     def __init__(
#         self,
#         goal_position=None,
#         size=10,
#         agent_start_pos=None,
#         agent_start_dir=None,
#         max_steps: int | None = None,
#         **kwargs,
#     ):
#         self.goal_position = goal_position
#         self.agent_start_pos = agent_start_pos
#         self.agent_start_dir = agent_start_dir

#         mission_space = MissionSpace(mission_func=self._gen_mission)

#         if max_steps is None:
#             max_steps = 4 * size**2

#         super().__init__(
#             mission_space=mission_space,
#             grid_size=size,
#             see_through_walls=True,
#             max_steps=max_steps,
#             **kwargs,
#         )

#     @staticmethod
#     def _gen_mission():
#         return "reach the goal"

#     def _gen_grid(self, width, height):
#         # Create an empty grid
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.wall_rect(0, 0, width, height)

#         # Generate vertical separation wall with opening at i==2
#         for i in range(0, height):
#             if i == 2:
#                 continue
#             else:
#                 self.grid.set(5, i, Wall())

#         # Place the agent first
#         if self.agent_start_pos is not None:
#             self.agent_pos = self.agent_start_pos
#             self.agent_dir = self.agent_start_dir
#         else:
#             self.place_agent()

#         # Determine goal position
#         if self.goal_position is None:
#             # Random position avoiding agent position
#             while True:
#                 goal_x = self._rand_int(1, width - 2)
#                 goal_y = self._rand_int(1, height - 2)
#                 if (goal_x, goal_y) != self.agent_pos:
#                     break
#         else:
#             goal_x, goal_y = self.goal_position

#         self.put_obj(Goal(), goal_x, goal_y)

#         self.mission = "reach the goal"


# class ParametricPutNearEnv(MiniGridEnv):
#     """
#     Environment with a parametric put near task.
#     Parameters: target_obj_pos, obj_to_move_pos, target_location - tuples (None for random)
#     """
#     def __init__(
#         self,
#         target_obj_pos=None,
#         obj_to_move_pos=None,
#         target_location=None,
#         size=10,
#         agent_start_pos=None,
#         agent_start_dir=None,
#         max_steps: int | None = None,
#         **kwargs,
#     ):
#         self.target_obj_pos = target_obj_pos
#         self.obj_to_move_pos = obj_to_move_pos
#         self.target_location = target_location
#         self.agent_start_pos = agent_start_pos
#         self.agent_start_dir = agent_start_dir

#         mission_space = MissionSpace(mission_func=self._gen_mission)

#         if max_steps is None:
#             max_steps = 4 * size**2

#         super().__init__(
#             mission_space=mission_space,
#             grid_size=size,
#             see_through_walls=True,
#             max_steps=max_steps,
#             **kwargs,
#         )

#     @staticmethod
#     def _gen_mission():
#         return "pick up the box and put it near the target object"

#     def _gen_grid(self, width, height):
#         # Create an empty grid
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.wall_rect(0, 0, width, height)

#         # Generate vertical separation wall with opening at i==2
#         for i in range(0, height):
#             if i == 2:
#                 continue
#             else:
#                 self.grid.set(5, i, Wall())

#         # Place the agent first
#         if self.agent_start_pos is not None:
#             self.agent_pos = self.agent_start_pos
#             self.agent_dir = self.agent_start_dir
#         else:
#             self.place_agent()

#         # Randomize box colors
#         box_colors = ["red", "green", "blue", "purple", "yellow", "grey"]
#         target_color = self._rand_elem(box_colors)
#         move_color = self._rand_elem(box_colors)
#         while move_color == target_color:  # Ensure different colors
#             move_color = self._rand_elem(box_colors)

#         # Determine target object position
#         if self.target_obj_pos is None:
#             while True:
#                 target_x = self._rand_int(1, width - 2)
#                 target_y = self._rand_int(1, height - 2)
#                 if (target_x, target_y) != self.agent_pos:
#                     break
#         else:
#             target_x, target_y = self.target_obj_pos

#         # Place the target object
#         self.put_obj(Box(target_color), target_x, target_y)

#         # Determine object to move position
#         if self.obj_to_move_pos is None:
#             while True:
#                 obj_x = self._rand_int(1, width - 2)
#                 obj_y = self._rand_int(1, height - 2)
#                 if (obj_x, obj_y) != self.agent_pos and (obj_x, obj_y) != (target_x, target_y):
#                     break
#         else:
#             obj_x, obj_y = self.obj_to_move_pos

#         # Place the object to be moved
#         self.put_obj(Box(move_color), obj_x, obj_y)

#         # Determine target location (goal position)
#         if self.target_location is None:
#             while True:
#                 goal_x = self._rand_int(1, width - 2)
#                 goal_y = self._rand_int(1, height - 2)
#                 if (goal_x, goal_y) != self.agent_pos and (goal_x, goal_y) != (target_x, target_y) and (goal_x, goal_y) != (obj_x, obj_y):
#                     break
#         else:
#             goal_x, goal_y = self.target_location

#         # Place the goal
#         self.put_obj(Goal(), goal_x, goal_y)

#         self.mission = "pick up the box and put it near the target object"


    
if __name__ == "__main__":
    env = MetaSimpleEnvReachOneGoal(render_mode="human")
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()