"""Визуализация результатов."""


def plot_results(datacollector, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен")
        return

    d = datacollector.data
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Популяция
    axes[0, 0].plot(d['step'], d['population'], color='steelblue')
    axes[0, 0].set_title('Популяция агентов')
    axes[0, 0].set_xlabel('Шаг')

    # 2. Частоты стратегий (репликаторная динамика)
    axes[0, 1].plot(d['step'], d['freq_cooperators'],
                    label='Кооператоры', color='green')
    axes[0, 1].plot(d['step'],
                    [1 - f for f in d['freq_cooperators']],
                    label='Дефекторы', color='red', linestyle='--')
    axes[0, 1].set_title('Частота стратегий (репликаторная динамика)')
    axes[0, 1].set_xlabel('Шаг')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    # 3. Ресурс среды + катастрофы
    ax = axes[1, 0]
    ax.plot(d['step'], d['total_env_resource'], color='orange')
    ax.set_title('Суммарный ресурс среды')
    ax.set_xlabel('Шаг')
    # Отметки катастроф
    for i, active in enumerate(d['catastrophe_active']):
        if active:
            ax.axvspan(d['step'][i], d['step'][i] + 1,
                       color='red', alpha=0.15)

    # 4. Средний ресурс агентов + средний метаболизм
    ax2 = axes[1, 1]
    ax2.plot(d['step'], d['avg_agent_resource'], color='purple',
             label='Средний ресурс агента')
    ax2.set_xlabel('Шаг')
    ax2.set_title('Адаптация агентов')
    ax3 = ax2.twinx()
    ax3.plot(d['step'], d['avg_metabolism'], color='gray',
             linestyle=':', label='Средний метаболизм')
    ax3.set_ylabel('Метаболизм')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    else:
        plt.show()
