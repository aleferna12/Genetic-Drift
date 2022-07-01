import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# N populacional de indivíduos diploides que permanecerá constante durante a simulação
POP = 1000
ALLELE_FREQS = [1 / 3] * 3
# Quanto da população vai ser substituída em cada iteração
# Quanto menor for, mais tempo os alelos duram em média
# Uma SUB_RATE de 1 quer dizer que cada geração é formada inteiramente por novos indivíduos
SUB_RATE = 1
# Número máximo de gerações a observar (-1 quer dizer que a simulação só para quando houver fixação)
GENERATIONS = 200
N_RUNS = 1


def simulate(pop, allele_freqs, sub_rate, gens, n_runs):
    alleles = tuple(range(len(allele_freqs)))
    runs = []
    for _ in trange(n_runs):
        new_pop = np.array(alleles * (2 * pop // len(alleles)) + alleles[:2 * pop % len(alleles)])
        stats = [np.bincount(new_pop, minlength=len(alleles)) / (2 * pop)]
        while 1 not in stats[-1] and (gens == -1 or len(stats) < gens):
            # Graças ao equilíbrio de hardy-weinberg podemos simplificar a população a gametas
            # Os gametas são aleatoriamente amostrados baseado nas frequências da última geração
            # Presume-se que um mesmo gameta poderia ser amostrado diversas vezes (há repetição)
            # Equivalente a np.random.choice(new_pop, round(2 * pop * sub_rate), replace=True)
            descendents = np.random.choice(alleles, round(2 * pop * sub_rate), p=stats[-1])
            survivors = np.random.choice(new_pop, round(2 * pop * (1 - sub_rate)), replace=False)
            new_pop = np.concatenate((descendents, survivors))
            stats.append(np.bincount(new_pop, minlength=len(alleles)) / (2 * pop))
        runs.append(stats)
    return np.array(max(runs, key=len))


def main():
    stats = simulate(POP, ALLELE_FREQS, SUB_RATE, GENERATIONS, N_RUNS)
    fig, axes = plt.subplots(2)
    x = range(len(stats)) 
    axes[0].plot(x, np.where(stats != 0, stats, np.nan))
    axes[0].set_ylim((0, 1))
    axes[0].autoscale(tight=True, axis="x")
    axes[1].stackplot(x, *np.transpose(stats))
    axes[1].autoscale(tight=True)
    fig.suptitle(f"Pop: {POP} | "
                 f"Subs: {SUB_RATE} | "
                 f"Gens: {len(stats)}",
                 fontsize="medium",
                 fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()