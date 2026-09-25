import time
from environments.EgeExp1 import EgeExp1
from environments.EgeExp2 import EgeExp2
from environments.EgeExp3 import EgeExp3
from environments.EgeExp4 import EgeExp4
from environments.EgeExp5 import EgeExp5
from environments.EgeExp6 import EgeExp6
from environments.EgeExp7 import EgeExp7
from environments.EgeExp8 import EgeExp8
from momab_main import run_anytime_experiment

envs = [EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8]

def benchmark():
    total_time = 0
    print("Benchmarking 1 run of 5000 pulls for each environment...")
    for env_cls in envs:
        env = env_cls()
        start = time.time()
        run_anytime_experiment(
            num_runs=1,
            max_budget=5000,
            environment=env,
            algorithms=["APE"],
            results_file=None,
            write=False,
            step=50
        )
        end = time.time()
        elapsed = end - start
        total_time += elapsed
        print(f"{env_cls.__name__}: 1 run = {elapsed:.2f} seconds")
        
    print("-" * 50)
    print(f"Total time for 1 run across all 8 envs: {total_time:.2f} seconds")
    
    # Calculate for 100 runs
    total_100_runs_seconds = total_time * 100
    total_100_runs_hours = total_100_runs_seconds / 3600
    
    print(f"Estimated sequential time for 100 runs on all 8 envs: {total_100_runs_hours:.2f} hours")
    
    # Also estimate parallel execution locally
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    parallel_hours = total_100_runs_hours / n_cores
    print(f"Estimated parallel time (assuming {n_cores} cores used efficiently): {parallel_hours:.2f} hours")
    
if __name__ == "__main__":
    benchmark()
