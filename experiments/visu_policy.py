from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from joblib import Parallel, delayed

def evaluate_seed(seed):
    """Evaluate a single policy seed and return success count and seed number."""
    try:
        model = PPO.load(f'policies/bigger_head_gcrl_seed_{seed}')
        
        env = DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0, max_steps=50)))
        success = 0
        for _ in range(5000):
            done = False
            s, _= env.reset()
            while not done:
                a = model.predict(s, deterministic=False)[0] # stochastic because POMDP
                s, r, term, trunc, _ = env.step(a)
                if r>0:
                    success += 1
                done = term or trunc
        
        print(f"Seed {seed}: {success} successes")
        return success, seed
    except Exception as e:
        print(f"Error evaluating seed {seed}: {e}")
        return 0, seed

# Parallel evaluation of all seeds
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(evaluate_seed)(s) for s in range(70)
)

# Find the best result
max_success, max_seed = max(results, key=lambda x: x[0])

print(f"Best result: {max_success} successes with seed {max_seed}")