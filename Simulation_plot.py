import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  

masses = [170.0 + 0.5*i for i in range(21)]

for m in masses:
    mass_str = f"{m:.1f}"   # to ensure 170.0, 170.5, ..., 180.0

    csv_file = f"/home/sawini-jana/Documents/EEEC_histogram_{mass_str}.csv"
    png_file = f"/home/sawini-jana/Documents/simulation_results_{mass_str}.png"

    print(f"Processing {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        continue
    
    # Remove zero values to make it less cluttered 
    df = df[df["value"] > 0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(df["z1_center"],df["z2_center"],df["z3_center"],c=df["value"],cmap="viridis",s=10)

    ax.set_xlabel("ζ₁")
    ax.set_ylabel("ζ₂")
    ax.set_zlabel("ζ₃")

    plt.colorbar(sc, label="EEEC density")
    plt.title(f"Simulation Plot {mass_str}")

    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)  