import subprocess
import re
import matplotlib.pyplot as plt
import statistics

def run_benchmark_for_n(n_value, iterations):
    ref_cycles_list = []
    rvv_cycles_list = []
    
    for i in range(iterations):
        try:
            # giving n value as arg to run script
            result = subprocess.run(["bash", "run.bash", str(n_value)], 
                                    capture_output=True, text=True, check=True)
            output = result.stdout

            if "Verify RVV: OK" in output:
                ref_match = re.search(r"Cycles ref: (\d+)", output)
                rvv_match = re.search(r"Cycles RVV: (\d+)", output)
                
                if ref_match and rvv_match:
                    ref_cycles_list.append(int(ref_match.group(1)))
                    rvv_cycles_list.append(int(rvv_match.group(1)))
        except Exception as e:
            continue
            
    return ref_cycles_list, rvv_cycles_list

def plot_scaling_results(results):
    n_sizes = sorted(results.keys())
    speedups = [results[n]['speedup'] for n in n_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_sizes, speedups, marker='s', color='green', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Scalar Baseline')
    
    plt.xscale('log', base=2)
    plt.xlabel('Array Size (N)')
    plt.ylabel('Speedup Factor (Scalar / RVV)')
    plt.title('RVV Performance Scaling vs Array Size')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    #saving to results folder
    plt.savefig('results/scaling_results.png')
    plt.show()



def plot_pairwise_bar(data):
    n_sizes = sorted(data.keys())
    ref_cycles = [data[n]['ref'] for n in n_sizes]
    rvv_cycles = [data[n]['rvv'] for n in n_sizes]
    
    x = range(len(n_sizes))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar([p - width/2 for p in x], ref_cycles, width, label='Scalar Ref', color='blue')
    plt.bar([p + width/2 for p in x], rvv_cycles, width, label='RVV', color='orange')
    
    plt.yscale('log')
    plt.xlabel('Array Size (N)')
    plt.ylabel('Cycles (log scale)')
    plt.title('Cycle Comparison: Scalar vs RVV')
    plt.xticks(x, n_sizes)
    plt.legend()
    plt.grid(True, which="both", ls="-")
    #saving to results folder
    plt.savefig('results/pairwise_comparison.png')
    plt.show()

if __name__ == "__main__":
    test_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    K = 30 # Number of iterations
    
    final_data = {}

    print("Compiling the code\n") 
    subprocess.run(["bash", "complie.bash"], check=True)

    for n in test_sizes:
        print(f"Benchmarking N={n}...", end=" ", flush=True)
        ref, rvv = run_benchmark_for_n(n, K)
        
        if ref and rvv:
            avg_ref = statistics.mean(ref)
            avg_rvv = statistics.mean(rvv)
            speedup = avg_ref / avg_rvv
            final_data[n] = {'ref': avg_ref, 'rvv': avg_rvv, 'speedup': speedup}
            print(f"Speedup: {speedup:.2f}")
        else:
            print("Failed, try again")

    plot_scaling_results(final_data)
    plot_pairwise_bar(final_data)