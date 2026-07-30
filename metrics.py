import os
import re
import argparse
import json
from statistics import mean, stdev

# Regular expressions to match lines in results.txt
# Supports both BASE and FINAL metrics, and old format for backward compat
iter_avg_pattern_base = re.compile(
    r"\[ITER (\d+)\] test BASE\s+avg: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
iter_std_pattern_base = re.compile(
    r"\[ITER (\d+)\] test BASE\s+std: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
iter_avg_pattern_final = re.compile(
    r"\[ITER (\d+)\] test FINAL avg: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
iter_std_pattern_final = re.compile(
    r"\[ITER (\d+)\] test FINAL std: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
# Old format fallback
iter_avg_pattern_old = re.compile(
    r"\[ITER (\d+)\] Evaluating test avg: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
iter_std_pattern_old = re.compile(
    r"\[ITER (\d+)\] Evaluating test std: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)

def _parse_matches(avg_matches, std_matches):
    """Parse matched tuples into per-iteration dict."""
    data = {}
    for avg_match, std_match in zip(avg_matches, std_matches):
        iter_num_avg, l1_avg, psnr_avg, lpips_avg, ssim_avg, mssim_avg = avg_match
        iter_num_std, l1_std, psnr_std, lpips_std, ssim_std, mssim_std = std_match
        assert iter_num_avg == iter_num_std, "Mismatch in iteration numbers for avg and std lines."
        iter_num = int(iter_num_avg)
        data[iter_num] = {
            "L1": (float(l1_avg), float(l1_std)),
            "PSNR": (float(psnr_avg), float(psnr_std)),
            "LPIPS": (float(lpips_avg), float(lpips_std)),
            "SSIM": (float(ssim_avg), float(ssim_std)),
            "MSSIM": (float(mssim_avg), float(mssim_std)),
        }
    return data

def extract_metrics(base_dir):
    all_iterations = set()
    run_metrics_base = []
    run_metrics_final = []

    for subdir in os.listdir(base_dir):
        results_path = os.path.join(base_dir, subdir, "results.txt")
        if not os.path.isfile(results_path):
            continue
        with open(results_path, "r") as file:
            content = file.read()

        # Try new format first
        avg_base = iter_avg_pattern_base.findall(content)
        std_base = iter_std_pattern_base.findall(content)
        avg_final = iter_avg_pattern_final.findall(content)
        std_final = iter_std_pattern_final.findall(content)

        if avg_base and std_base:
            base_data = _parse_matches(avg_base, std_base)
        else:
            # Fallback to old format
            avg_old = iter_avg_pattern_old.findall(content)
            std_old = iter_std_pattern_old.findall(content)
            base_data = _parse_matches(avg_old, std_old) if avg_old and std_old else {}

        final_data = _parse_matches(avg_final, std_final) if avg_final and std_final else {}

        for it in base_data:
            all_iterations.add(it)
        for it in final_data:
            all_iterations.add(it)

        run_metrics_base.append(base_data)
        run_metrics_final.append(final_data if final_data else base_data)

    all_iterations = sorted(all_iterations)
    return all_iterations, run_metrics_base, run_metrics_final

def _aggregate(run_metrics_list, all_iterations):
    result = {}
    for iter_num in all_iterations:
        values = {"PSNR": [], "LPIPS": [], "SSIM": [], "MSSIM": [], "L1": []}
        for run_data in run_metrics_list:
            available = sorted(k for k in run_data if k <= iter_num)
            if available:
                m = run_data[available[-1]]
                for k in values:
                    values[k].append(m[k][0])
        if values["PSNR"]:
            result[iter_num] = {}
            for k in values:
                vals = values[k]
                result[iter_num][k] = (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)
    return result

def save_results(base_dir, all_iterations, run_metrics_base, run_metrics_final):
    base_agg = _aggregate(run_metrics_base, all_iterations)
    final_agg = _aggregate(run_metrics_final, all_iterations)
    has_final = any(final_agg.get(it, {}).get("PSNR", (0,0)) != base_agg.get(it, {}).get("PSNR", (0,0))
                    for it in all_iterations)

    results_path = os.path.join(base_dir, "mean_results.txt")
    with open(results_path, "w") as file:
        file.write("Mean and Std Values per Iteration:\n")

        for metric_name in ["PSNR", "LPIPS", "SSIM", "MSSIM", "L1"]:
            file.write(f"\n{metric_name} per Iteration (BASE):\n")
            for it in all_iterations:
                if it in base_agg and metric_name in base_agg[it]:
                    m, s = base_agg[it][metric_name]
                    file.write(f"Iteration {it}: Mean = {m:.4f}, Std Dev = {s:.4f}\n")

            if has_final:
                file.write(f"\n{metric_name} per Iteration (FINAL = base+spec):\n")
                for it in all_iterations:
                    if it in final_agg and metric_name in final_agg[it]:
                        m, s = final_agg[it][metric_name]
                        file.write(f"Iteration {it}: Mean = {m:.4f}, Std Dev = {s:.4f}\n")

    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and compute mean and std values from results.txt files.")
    parser.add_argument("base_dir", type=str, help="Base directory containing subfolders with results.txt files.")
    args = parser.parse_args()

    all_iterations, run_metrics_base, run_metrics_final = extract_metrics(args.base_dir)
    save_results(args.base_dir, all_iterations, run_metrics_base, run_metrics_final)
