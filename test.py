def linear_annealing(start, end, total_steps, now_step):
    return max(end, start - (start - end) * (now_step / total_steps))

for i in range(1,100):
    out = linear_annealing(1.0, 0.0, 100, i)
    print(f"Step {i}: {out}")