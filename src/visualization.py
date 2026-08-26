import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

IMITATION_TYPES = ["none", "best_neighbor", "pairwise_diff", "proportional_m", "fermi_m"]

IMITATION_LABELS = {
    "none": "Консерватор (нет имитации)",
    "best_neighbor": "Подражатель лучшего",
    "pairwise_diff": "Пропорц. (Репликатор)",
    "proportional_m": "Моран-имитатор",
    "fermi_m": "Ферми/Логит",
}

IMITATION_COLORS = {
    "none": "#888888",
    "best_neighbor": "#e74c3c",
    "pairwise_diff": "#3498db",
    "proportional_m": "#2ecc71",
    "fermi_m": "#9b59b6",
}


def plot_results(df: pd.DataFrame, save_path=None):
    """Комплексная визуализация: 3x2 сетка графиков."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # === 1. Популяция + катастрофы + Группы ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Step'], df['Population'], color='steelblue', linewidth=1.5, label='Популяция')
    ax1.set_title('Популяция агентов и Групповая динамика', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Шаг')
    ax1.set_ylabel('Число агентов', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    if 'Alive_Groups' in df.columns and df['Alive_Groups'].max() > 1:
        ax1b = ax1.twinx()
        ax1b.plot(df['Step'], df['Alive_Groups'], color='darkgreen', linestyle=':', linewidth=1.5, label='Живых групп')
        ax1b.set_ylabel('Количество групп', color='darkgreen')
        ax1b.tick_params(axis='y', labelcolor='darkgreen')
        ax1b.legend(loc='upper right', fontsize=8)
        
    for i, active in enumerate(df.get('Catastrophe_Active', [])):
        if active:
            ax1.axvspan(df['Step'].iloc[i], df['Step'].iloc[i] + 1,
                        color='red', alpha=0.12)
    ax1.legend(loc='upper left', fontsize=8)

    # === 2. Частоты стратегий ===
    ax2 = fig.add_subplot(gs[0, 1])
    for s, color in [("AlwaysC", "green"), ("TFT", "limegreen"),
                     ("GTFT", "teal"), ("WSLS", "orange"), ("AlwaysD", "red")]:
        col = f"Freq_{s}"
        if col in df.columns:
            ax2.plot(df['Step'], df[col], label=s, color=color, linewidth=1.2)
    ax2.set_title('Частоты игровых стратегий', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Шаг')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8, ncol=2)

    # === 3. Частоты типов имитаторов (КЛЮЧЕВОЙ ГРАФИК) ===
    ax3 = fig.add_subplot(gs[1, 0])
    for t in IMITATION_TYPES:
        col = f"ImitFreq_{t}"
        if col in df.columns:
            ax3.plot(df['Step'], df[col],
                     label=IMITATION_LABELS.get(t, t),
                     color=IMITATION_COLORS.get(t, "gray"),
                     linewidth=1.5)
    ax3.set_title('🧬 Эволюция типов социального обучения',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Шаг')
    ax3.set_ylabel('Доля в популяции')
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(fontsize=8)

    # === 4. Средний ресурс по типам имитаторов (приспособленность) ===
    ax4 = fig.add_subplot(gs[1, 1])
    for t in IMITATION_TYPES:
        col = f"ImitAvgResource_{t}"
        if col in df.columns:
            # Сглаживаем для читаемости
            smoothed = df[col].rolling(window=max(1, len(df)//50), min_periods=1).mean()
            ax4.plot(df['Step'], smoothed,
                     label=IMITATION_LABELS.get(t, t),
                     color=IMITATION_COLORS.get(t, "gray"),
                     linewidth=1.5)
    ax4.set_title('💰 Средний ресурс по типам имитаторов\n(приспособленность)',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Шаг')
    ax4.set_ylabel('Средний ресурс')
    ax4.legend(fontsize=8)

    # === 5. Среда ===
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df['Step'], df['Total_Env_Resource'], color='orange', linewidth=1.2)
    ax5.set_title('Суммарный ресурс среды', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Шаг')
    for i, active in enumerate(df.get('Catastrophe_Active', [])):
        if active:
            ax5.axvspan(df['Step'].iloc[i], df['Step'].iloc[i] + 1,
                        color='red', alpha=0.15)

    # === 6. Эволюция интенсивности имитации (параметр m) ===
    ax6 = fig.add_subplot(gs[2, 1])
    if 'Avg_Imitation_Intensity' in df.columns:
        ax6.plot(df['Step'], df['Avg_Imitation_Intensity'],
                 color='purple', linewidth=1.5, label='Средний m (интенсивность)')
    if 'Avg_Imitation_Rate' in df.columns:
        ax6b = ax6.twinx()
        ax6b.plot(df['Step'], df['Avg_Imitation_Rate'],
                  color='brown', linestyle='--', linewidth=1.2, label='Средний rate')
        ax6b.set_ylabel('Imitation Rate', color='brown')
        ax6b.legend(loc='upper right', fontsize=8)
    ax6.set_title('⚙️ Эволюция параметров имитации',
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('Шаг')
    ax6.set_ylabel('Интенсивность (m)', color='purple')
    ax6.legend(loc='upper left', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    else:
        plt.show()


def plot_imitation_comparison(df: pd.DataFrame, save_path=None):
    """
    Дополнительный график: финальное сравнение успешности типов имитаторов.
    Столбчатая диаграмма — средняя приспособленность каждого типа.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Левый: финальные частоты
    ax = axes[0]
    types = []
    freqs = []
    colors = []
    for t in IMITATION_TYPES:
        col = f"ImitFreq_{t}"
        if col in df.columns:
            types.append(IMITATION_LABELS.get(t, t))
            freqs.append(df[col].iloc[-1])
            colors.append(IMITATION_COLORS.get(t, "gray"))

    bars = ax.barh(types, freqs, color=colors)
    ax.set_xlabel('Доля в популяции (финальная)')
    ax.set_title('🏆 Финальное распределение типов имитаторов', fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, freq in zip(bars, freqs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{freq:.1%}', va='center', fontsize=9)

    # Правый: средний ресурс (приспособленность)
    ax2 = axes[1]
    resources = []
    for t in IMITATION_TYPES:
        col = f"ImitAvgResource_{t}"
        if col in df.columns:
            # Среднее за последнюю четверть симуляции
            quarter = len(df) // 4
            resources.append(df[col].iloc[-quarter:].mean())
        else:
            resources.append(0)

    bars2 = ax2.barh(types, resources, color=colors)
    ax2.set_xlabel('Средний ресурс (последняя четверть)')
    ax2.set_title('💪 Средняя приспособленность', fontweight='bold')
    for bar, res in zip(bars2, resources):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{res:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сравнительный график сохранён: {save_path}")
    else:
        plt.show()
