import csv

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)

HARDWARE_PEAK = 14000  # 14TF max for A100D

FILES = [
    "baseline",
    "cublas",

    "baseline_m8n8k8",
    "baseline_m8n8k8_128b",

    "baseline_m16n16k16_128b",
    "baseline_m16n16k16_128b_4W",

    #"baseline_m32n32k32_128b",
    #"baseline_m32n32k32_128b_4W",
    "baseline_m64n64k32_128b",
    "baseline_m64n64k32_128b_4W",
    "baseline_m64n64k32_128b_4W_CG",

    #"baseline_m64n64k64_128b",
    #"baseline_m64n64k64_128b_4W",
    "baseline_m128n64k16_128b_4W",
    "baseline_m128n64k16_128b_4W_CG",
    "baseline_m128n128k16_128b_4W",
]

y_min = float("inf")
x_min = float("inf")
x_max = float("-inf")
xticks_set = set()

for file_name in FILES:
    xs = []
    ys = []
    with open(f"{file_name}.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            m, n, k = row["M"], row["N"], row["K"]
            gflops = row["GFLOPS"]
            assert (
                m == n and n == k
            )  # only support (SIZE,SIZE) @ (SIZE,SIZE) = (SIZE,SIZE)
            xs.append(int(m))
            ys.append(float(gflops))
            xticks_set.add(int(m))
    x_min = min(x_min, min(xs))
    x_max = max(x_max, max(xs))
    y_min = min(y_min, min(ys))
    plt.plot(xs, ys, label=file_name, marker="x")

plt.xlim(x_min, x_max)
plt.ylim(0, HARDWARE_PEAK * 1.50)
plt.xscale("log", base=2)
# plt.yscale("log", base=2)
plt.hlines(
    y=HARDWARE_PEAK,
    xmin=x_min,
    xmax=x_max,
    colors="red",
    linestyles="dashed",
    label="hardware peak",
)
plt.ylabel("Performance (GFLOP/s)")
plt.xlabel("Square matrices size (M=N=K)")
plt.legend()
xticks = sorted(list(xticks_set))
# plt.axes().set_xticks(xticks)
plt.xticks(xticks, xticks)
plt.savefig("bench.png", dpi=600)
