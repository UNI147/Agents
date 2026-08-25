import pandas as pd
import matplotlib.pyplot as plt

def plot_results(df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(df['Step'], df['Population'], color='steelblue')
    axes[0, 0].set_title('Популяция агентов')
    axes[0, 0].set_xlabel('Шаг')

    axes[0, 1].plot(df['Step'], df['Freq_Cooperators'], label='Кооператоры', color='green')
    axes[0, 1].plot(df['Step'], 1 - df['Freq_Cooperators'], label='Дефекторы', color='red', linestyle='--')
    axes[0, 1].set_title('Частота стратегий')
    axes[0, 1].set_xlabel('Шаг')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    ax = axes[1, 0]
    ax.plot(df['Step'], df['Total_Env_Resource'], color='orange')
    ax.set_title('Суммарный ресурс среды')
    ax.set_xlabel('Шаг')
    for i, active in enumerate(df['Catastrophe_Active']):
        if active:
            ax.axvspan(df['Step'].iloc[i], df['Step'].iloc[i] + 1, color='red', alpha=0.15)

    ax2 = axes[1, 1]
    ax2.plot(df['Step'], df['Avg_Agent_Resource'], color='purple', label='Средний ресурс агента')
    ax2.set_xlabel('Шаг')
    ax2.set_title('Адаптация агентов')
    ax3 = ax2.twinx()
    ax3.plot(df['Step'], df['Avg_Metabolism'], color='gray', linestyle=':', label='Средний метаболизм')
    ax3.set_ylabel('Метаболизм')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    else:
        plt.show()
