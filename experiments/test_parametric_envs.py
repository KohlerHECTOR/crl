#!/usr/bin/env python3
"""
Test script for the three parametric environments:
1. ParametricDoorKeyEnv - parametric key position
2. ParametricGoalReachEnv - parametric goal position  
3. ParametricPutNearEnv - parametric object positions and target location
"""

from minigrid_envs import ParametricDoorKeyEnv, ParametricGoalReachEnv, ParametricPutNearEnv
from minigrid.manual_control import ManualControl
import numpy as np

def test_parametric_door_key():
    """Test the parametric door key environment with different key positions"""
    print("Testing ParametricDoorKeyEnv...")
    
    # Test with different key positions (adjusted for size 10 grid)
    # Note: key will be placed on agent's side automatically
    key_positions = [(2, 2), (3, 3), (2, 7), (3, 8)]
    
    for i, key_pos in enumerate(key_positions):
        print(f"  Test {i+1}: Key position {key_pos}")
        env = ParametricDoorKeyEnv(
            key_position=key_pos,
            render_mode="human"
        )
        
        # Enable manual control for testing
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
        
        # Close the environment
        env.close()
    
    # Test with random key placement
    print("  Test 5: Random key placement")
    env = ParametricDoorKeyEnv(
        key_position=None,  # Will be placed randomly on agent's side
        render_mode="human"
    )
    
    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    
    # Close the environment
    env.close()

def test_parametric_goal_reach():
    """Test the parametric goal reach environment with different goal positions"""
    print("Testing ParametricGoalReachEnv...")
    
    # Test with different goal positions (adjusted for size 10 grid)
    goal_positions = [(3, 3), (7, 3), (3, 7), (7, 7)]
    
    for i, goal_pos in enumerate(goal_positions):
        print(f"  Test {i+1}: Goal position {goal_pos}")
        env = ParametricGoalReachEnv(
            goal_position=goal_pos,
            render_mode="human"
        )
        
        # Enable manual control for testing
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
        
        # Close the environment
        env.close()
    
    # Test with random goal placement
    print("  Test 5: Random goal placement")
    env = ParametricGoalReachEnv(
        goal_position=None,  # Will be placed randomly
        render_mode="human"
    )
    
    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    
    # Close the environment
    env.close()

def test_parametric_put_near():
    """Test the parametric put near environment with different configurations"""
    print("Testing ParametricPutNearEnv...")
    
    # Test with different configurations (adjusted for size 10 grid)
    configurations = [
        ((2, 2), (4, 4), (8, 8)),  # target_obj, obj_to_move, target_location
        ((1, 1), (6, 6), (8, 8)),
        ((3, 1), (1, 6), (7, 8)),
    ]
    
    for i, (target_obj, obj_to_move, target_location) in enumerate(configurations):
        print(f"  Test {i+1}: Target obj {target_obj}, Object to move {obj_to_move}, Target location {target_location}")
        env = ParametricPutNearEnv(
            target_obj_pos=target_obj,
            obj_to_move_pos=obj_to_move,
            target_location=target_location,
            render_mode="human"
        )
        
        # Enable manual control for testing
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
        
        # Close the environment
        env.close()
    
    # Test with random placement
    print("  Test 4: Random placement")
    env = ParametricPutNearEnv(
        target_obj_pos=None,  # Will be placed randomly
        obj_to_move_pos=None,  # Will be placed randomly
        target_location=None,  # Will be placed randomly
        render_mode="human"
    )
    
    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    
    # Close the environment
    env.close()

def test_programmatic_usage():
    """Test programmatic usage of the environments without manual control"""
    print("Testing programmatic usage...")
    
    # Test ParametricDoorKeyEnv
    env = ParametricDoorKeyEnv(key_position=None)  # Random placement
    obs, info = env.reset()
    print(f"DoorKeyEnv observation shape: {obs['image'].shape}")
    env.close()
    
    # Test ParametricGoalReachEnv
    env = ParametricGoalReachEnv(goal_position=None)  # Random placement
    obs, info = env.reset()
    print(f"GoalReachEnv observation shape: {obs['image'].shape}")
    env.close()
    
    # Test ParametricPutNearEnv
    env = ParametricPutNearEnv(
        target_obj_pos=None,  # Random placement
        obj_to_move_pos=None,  # Random placement
        target_location=None   # Random placement
    )
    obs, info = env.reset()
    print(f"PutNearEnv observation shape: {obs['image'].shape}")
    env.close()

if __name__ == "__main__":
    print("Testing Parametric Environments")
    print("=" * 40)
    
    # Uncomment the following lines to test with manual control
    test_parametric_door_key()
    test_parametric_goal_reach()
    test_parametric_put_near()
    
    # Test programmatic usage
    test_programmatic_usage()
    
    print("\nAll tests completed!")
    print("\nTo test with manual control, uncomment the test functions in the main block.")
