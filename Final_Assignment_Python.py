# -*- coding: utf-8 -*-
"""
Created on Tue Apr 7 18:05:19 2026

@author: sacha
"""

import time
import random
import math
import numpy as np
from functools import lru_cache

NUM_RUNS = 1000
ARRAY_SIZE = 500000

tasks = [
    {"id": 1, "C": 1, "T": 10, "D": 10, "jobs": 8},
    {"id": 2, "C": 3, "T": 10, "D": 10, "jobs": 8},
    {"id": 3, "C": 2, "T": 20, "D": 20, "jobs": 4},
    {"id": 4, "C": 2, "T": 20, "D": 20, "jobs": 4},
    {"id": 5, "C": 2, "T": 40, "D": 40, "jobs": 2},
    {"id": 6, "C": 2, "T": 40, "D": 40, "jobs": 2},
    {"id": 7, "C": 3, "T": 80, "D": 80, "jobs": 1},
]


def random_large_number():
    a = random.randint(0, 99999)
    b = random.randint(0, 99999)
    return a * 100000 + b


def measure_tau1():
    execution_times = []
    result = 0

    for _ in range(NUM_RUNS):
        x = random_large_number()
        y = random_large_number()

        start = time.perf_counter()
        for _ in range(ARRAY_SIZE):
            result = x * y
            x += 1
            y += 1
        end = time.perf_counter()

        execution_times.append(end - start)

    execution_times = np.array(execution_times)

    print("=== Part 1: tau_1 execution time statistics ===")
    print(f"Min  = {np.min(execution_times):.9f} seconds")
    print(f"Q1   = {np.percentile(execution_times, 25):.9f} seconds")
    print(f"Q2   = {np.percentile(execution_times, 50):.9f} seconds")
    print(f"Q3   = {np.percentile(execution_times, 75):.9f} seconds")
    print(f"Max  = {np.max(execution_times):.9f} seconds")
    print(f"WCET = {np.max(execution_times):.9f} seconds")
    print(f"Last result = {result}\n")


def lcm(a, b):
    return a * b // math.gcd(a, b)


def compute_hyperperiod():
    hp = tasks[0]["T"]
    for t in tasks[1:]:
        hp = lcm(hp, t["T"])
    return hp


def compute_utilization():
    return sum(t["C"] / t["T"] for t in tasks)


def print_task_set():
    print("=== Part 2: Task set ===")
    print("Task\tC\tT\tD\tJobs in [0,H)")
    for t in tasks:
        print(f"tau_{t['id']}\t{t['C']}\t{t['T']}\t{t['D']}\t{t['jobs']}")
    print()


def print_job_release_list():
    print("=== Part 3: Jobs in one hyperperiod ===")
    print("Idx\tTask\tJob\tRelease\tExec\tDeadline")

    jobs = []

    for t in tasks:
        for j in range(t["jobs"]):
            release = j * t["T"]
            deadline = release + t["D"]
            jobs.append({
                "task_id": t["id"],
                "job_id": j + 1,
                "release": release,
                "exec": t["C"],
                "deadline": deadline
            })

    jobs.sort(key=lambda x: (x["release"], x["deadline"], x["task_id"]))

    for idx, job in enumerate(jobs):
        print(
            f"{idx}\ttau_{job['task_id']}\t{job['job_id']}\t"
            f"{job['release']}\t{job['exec']}\t{job['deadline']}"
        )

    print(f"\nTotal jobs = {len(jobs)}\n")


def all_done(counts):
    return all(counts[i] == tasks[i]["jobs"] for i in range(len(tasks)))


def next_release_after_now(counts, now):
    candidates = []

    for i, t in enumerate(tasks):
        if counts[i] < t["jobs"]:
            release = counts[i] * t["T"]
            if release > now:
                candidates.append(release)

    return min(candidates) if candidates else None


def solve_optimal(allow_tau5_miss=False):
    @lru_cache(maxsize=None)
    def dp(counts, now):
        if all_done(counts):
            return 0, []

        best_cost = float("inf")
        best_plan = None

        for i, t in enumerate(tasks):
            if counts[i] >= t["jobs"]:
                continue

            release = counts[i] * t["T"]
            deadline = release + t["D"]
            finish = now + t["C"]

            if release > now:
                continue

            if not (allow_tau5_miss and t["id"] == 5) and finish > deadline:
                continue

            new_counts = list(counts)
            new_counts[i] += 1
            sub_cost, sub_plan = dp(tuple(new_counts), finish)

            waiting = now - release
            total_cost = waiting + sub_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_plan = [{
                    "task_id": t["id"],
                    "job_id": counts[i] + 1,
                    "release": release,
                    "start": now,
                    "finish": finish,
                    "deadline": deadline,
                    "waiting": waiting,
                    "response": finish - release,
                    "missed": finish > deadline
                }] + sub_plan

        nr = next_release_after_now(counts, now)
        if nr is not None:
            sub_cost, sub_plan = dp(counts, nr)
            if sub_cost < best_cost:
                best_cost = sub_cost
                best_plan = [{"idle_from": now, "idle_to": nr}] + sub_plan

        return best_cost, best_plan

    init_counts = tuple(0 for _ in tasks)
    return dp(init_counts, 0)


def summarize_schedule(title, plan, allow_tau5_miss=False):
    exec_jobs = [x for x in plan if "task_id" in x]
    idle_slots = [x for x in plan if "idle_from" in x]

    total_waiting = sum(j["waiting"] for j in exec_jobs)
    busy_time = sum(j["finish"] - j["start"] for j in exec_jobs)
    hyperperiod = compute_hyperperiod()
    idle_time = hyperperiod - busy_time
    
    total_missed = sum(1 for j in exec_jobs if j["missed"])
    missed_except_tau5 = sum(1 for j in exec_jobs if j["missed"] and j["task_id"] != 5)

    print(f"=== {title} ===")
    print("Task\tJob\tRelease\tStart\tFinish\tDeadline\tWait\tResponse\tMiss")

    for j in exec_jobs:
        miss = "YES" if j["missed"] else "NO"
        print(
            f"tau_{j['task_id']}\t{j['job_id']}\t{j['release']}\t{j['start']}\t"
            f"{j['finish']}\t{j['deadline']}\t\t{j['waiting']}\t{j['response']}\t\t{miss}"
        )

    print(f"\nTotal waiting time = {total_waiting}")
    print(f"Processor idle time = {idle_time}")
    print(f"Total missed deadlines = {total_missed}")

    if not allow_tau5_miss:
        print("Schedulable: YES\n" if total_missed == 0 else "Schedulable: NO\n")
    else:
        print(f"Missed deadlines excluding tau_5 = {missed_except_tau5}")
        ok = "YES" if missed_except_tau5 == 0 else "NO"
        print(f"All non-tau_5 deadlines are respected: {ok}\n")


def main():
    random.seed()

    hyperperiod = compute_hyperperiod()

    measure_tau1()
    print_task_set()

    print("=== Part 2B: Global properties ===")
    print(f"Hyperperiod = {hyperperiod}")
    print(f"Utilization = {compute_utilization():.4f}\n")

    print_job_release_list()

    best_cost_no_miss, plan_no_miss = solve_optimal(allow_tau5_miss=False)
    if plan_no_miss is None:
        print("=== Part 4A: Optimal schedule with no missed deadlines ===")
        print("No feasible schedule found.\n")
    else:
        summarize_schedule("Part 4A: Optimal schedule with no missed deadlines", plan_no_miss, False)

    best_cost_tau5, plan_tau5 = solve_optimal(allow_tau5_miss=True)
    if plan_tau5 is None:
        print("=== Part 4B: Optimal schedule with tau_5 allowed to miss ===")
        print("No feasible schedule found.\n")
    else:
        summarize_schedule("Part 4B: Optimal schedule with tau_5 allowed to miss", plan_tau5, True)


if __name__ == "__main__":
    main()