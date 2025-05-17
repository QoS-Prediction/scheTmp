import concurrent.futures
import subprocess
import itertools
import datetime
from tqdm import tqdm

def run_simulation(case, pm_num, resch, is_mask, vm_ratio, is_mask_oracle, is_mask_dyn, max_dyn_mask_ratio, g_alpha, mask_ratio):
    # cmd = [
    #     "python", "simulation_dynamic.py", 
    #     "--case", str(case), 
    #     "--PMnum", str(pm_num), 
    #     "--vm_ratio", str(vm_ratio), 
    #     "--resch", str(resch), 
    #     "--is_mask", str(is_mask)
    # ]

    cmd = [
        "cpulimit", "-l", "300", "--", 
        "python", "simulation_dynamic.py", 
        "--case", str(case), 
        "--PMnum", str(pm_num), 
        "--vm_ratio", str(vm_ratio), 
        "--resch", str(resch), 
        "--is_mask", str(is_mask)
    ]
    if is_mask:
        cmd.extend(["--is_mask_oracle", str(is_mask_oracle)])
        if is_mask_oracle:
            cmd.extend(["--mask_ratio", str(mask_ratio)]) 
        else:
            cmd.extend(["--is_mask_dyn", str(is_mask_dyn)])
            if is_mask_dyn:
                cmd.extend(["--max_dyn_mask_ratio", str(max_dyn_mask_ratio)])  # is_mask_dyn=True，使用 max_dyn_mask_ratio
                cmd.extend(["--g_alpha", str(g_alpha)])
            else:
                cmd.extend(["--mask_ratio", str(mask_ratio)])  # is_mask_dyn=False，使用 mask_ratio
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"Simulation (case={case}, PMnum={pm_num}, resch={resch}, is_mask={is_mask}, vm_ratio={vm_ratio}, "
            f"is_mask_oracle={is_mask_oracle}, is_mask_dyn={is_mask_dyn}, max_dyn_mask_ratio={max_dyn_mask_ratio}, "
            f"g_alpha={g_alpha}, mask_ratio={mask_ratio}) completed.")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Simulation (case={case}, PMnum={pm_num}, resch={resch}, is_mask={is_mask}, vm_ratio={vm_ratio}, "
            f"is_mask_oracle={is_mask_oracle}, is_mask_dyn={is_mask_dyn}, max_dyn_mask_ratio={max_dyn_mask_ratio}, "
            f"g_alpha={g_alpha}, mask_ratio={mask_ratio}) failed.\n"
            f"Return code: {e.returncode}\n"
            f"Standard Output:\n{e.stdout}\n"
            f"Standard Error:\n{e.stderr}\n"
        )
        print(error_message)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"simulation_errors_{timestamp}.log", "a") as f:
            f.write(error_message + "\n" + "=" * 80 + "\n")


cases = range(10)
pm_nums = [10]
vm_ratios = [3]
resch_methods = ['PEAS', 'FFD']  # ['RIAL', 'FFD', 'PEAS']  
is_masks = [True]  # [False, True]
is_mask_oracles = [False]
is_mask_dyns = [True]
g_alphas = [round(i * 0.1, 1) for i in range(1, 11)]
max_dyn_mask_ratios = [0.8]
mask_ratios = [0.5]  # [round(i * 0.1, 1) for i in range(1, 10)]

base_combinations = list(itertools.product(cases, pm_nums, resch_methods, is_masks, vm_ratios))
param_combinations = []
for case, pm_num, resch, is_mask, vm_ratio in base_combinations:
    if not is_mask:
        param_combinations.append((case, pm_num, resch, is_mask, vm_ratio, None, 0, 0, 0, 0))
    else:
        for is_mask_oracle in is_mask_oracles:
            if is_mask_oracle:
                for mask_ratio in mask_ratios:
                    param_combinations.append((case, pm_num, resch, is_mask, vm_ratio, is_mask_oracle, 0, 0, 0, mask_ratio))
            else:        
                for is_mask_dyn in is_mask_dyns:
                    if is_mask_dyn:
                        for max_dyn_mask_ratio, g_alpha in itertools.product(max_dyn_mask_ratios, g_alphas):
                            param_combinations.append((case, pm_num, resch, is_mask, vm_ratio, \
                                                       is_mask_oracle, is_mask_dyn, max_dyn_mask_ratio, g_alpha, 0))
                    else:
                        for mask_ratio in mask_ratios:
                            param_combinations.append((case, pm_num, resch, is_mask, vm_ratio, \
                                                       is_mask_oracle, is_mask_dyn, 0, 0, mask_ratio))
print(f"Total parameter combinations: {len(param_combinations)}")

# 并行执行
max_workers = 10
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_simulation, *params): params for params in param_combinations}
    with tqdm(total=len(futures), desc="Running simulations") as pbar:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 执行任务
            except Exception as e:
                print(f"Error in {futures[future]}: {e}")  # 处理异常
            finally:
                pbar.update(1)  # 更新进度条

print("All simulations completed.")
# pkill -9 -f simulation_dynamic.py
