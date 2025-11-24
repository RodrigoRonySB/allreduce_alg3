# plot_any.py
import sys, csv
import matplotlib.pyplot as plt

def load_csv(path):
    xs = []
    algo = []          # alg3 / alg5
    native = []
    zero = None        # solo se usará si existe la columna
    algo_label = None

    with open(path, 'r') as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames or []

        # --- columna X ---
        x_key = None
        for key in ('datasize_bytes_per_proc', 'datasize_bytes'):
            if key in fields:
                x_key = key
                break
        if x_key is None:
            raise KeyError("No 'datasize_bytes_per_proc' or 'datasize_bytes' column in CSV.")

        # --- qué algoritmo principal hay (alg5 o alg3) ---
        if 'best_us_alg5' in fields:
            algo_key = 'best_us_alg5'
            algo_label = 'Algorithm 5'
        elif 'best_us_alg3' in fields:
            algo_key = 'best_us_alg3'
            algo_label = 'Algorithm 3'
        else:
            raise KeyError("No 'best_us_alg5' or 'best_us_alg3' column in CSV.")

        # --- ¿hay columna zero-copy? ---
        has_zero = 'best_us_zero' in fields
        if has_zero:
            zero = []

        # --- columna native obligatoria ---
        if 'best_us_native' not in fields:
            raise KeyError("No 'best_us_native' column in CSV.")

        # --- leer filas ---
        for row in rdr:
            x = float(row[x_key])
            xs.append(x)

            algo.append(float(row[algo_key]))
            native.append(float(row['best_us_native']))

            if has_zero:
                zero.append(float(row['best_us_zero']))

    return xs, algo, native, algo_label, zero

def main():
    if len(sys.argv) < 3:
        print("Uso: python3 plot_any.py <csv1> <titulo1> [<csv2> <titulo2>] [out.png]")
        sys.exit(1)

    csv1, title1 = sys.argv[1], sys.argv[2]
    csv2 = title2 = None
    out = "plot.png"

    # parsing sencillo de args opcionales
    if len(sys.argv) >= 5 and not sys.argv[3].endswith(".png"):
        csv2, title2 = sys.argv[3], sys.argv[4]
        if len(sys.argv) >= 6 and sys.argv[5].endswith(".png"):
            out = sys.argv[5]
    elif sys.argv[-1].endswith(".png"):
        out = sys.argv[-1]

    plt.figure(figsize=(10, 4.5))
    panels = 2 if csv2 else 1

    for idx, (path, ttl) in enumerate(((csv1, title1), (csv2, title2))):
        if not path:
            break

        x, y_algo, y_nat, algo_label, y_zero = load_csv(path)

        if panels == 2:
            plt.subplot(1, 2, idx + 1)

        # Native
        plt.loglog(x, y_nat, marker='_', linestyle='-', label='MPI_Allreduce nativo')

        # Alg principal (Alg3 / Alg5)
        plt.loglog(x, y_algo, marker='x', linestyle='-', label=algo_label)

        # Zero-copy (si existe en el CSV)
        if y_zero is not None:
            plt.loglog(x, y_zero, marker='o', linestyle='-', label='Zero-copy')

        plt.xlabel('Datasize (Bytes) per process')
        plt.ylabel('Time (microseconds)')
        plt.title(ttl)
        plt.grid(True, which='both', ls=':', alpha=0.4)
        plt.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Guardado: {out}")

if __name__ == "__main__":
    main()
