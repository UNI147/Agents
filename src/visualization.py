import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

IMITATION_TYPES = ["none", "best_neighbor", "pairwise_diff", "proportional_m", "fermi_m"]
IMITATION_LABELS = {
    "none": "Консерватор (нет имитации)", "best_neighbor": "Подражатель лучшего",
    "pairwise_diff": "Пропорц. (Репликатор)", "proportional_m": "Моран-имитатор", "fermi_m": "Ферми/Логит",
}
IMITATION_COLORS = {
    "none": "#888888", "best_neighbor": "#e74c3c", "pairwise_diff": "#3498db",
    "proportional_m": "#2ecc71", "fermi_m": "#9b59b6",
}

def plot_results(df: pd.DataFrame, save_path=None):
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

    # === 1. ПОПУЛЯЦИЯ + ГРУППЫ (Возвращено как было) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Step'], df['Population'], color='steelblue', linewidth=1.5, label='Популяция')
    
    if 'Alive_Groups' in df.columns and df['Alive_Groups'].max() > 1:
        ax1_g = ax1.twinx()
        ax1_g.plot(df['Step'], df['Alive_Groups'], color='darkred', linestyle='--', linewidth=1.5, label='Живых групп')
        ax1_g.set_ylabel('Количество групп', color='darkred')
        ax1_g.tick_params(axis='y', labelcolor='darkred')
        
    ax1.set_title('Популяция агентов и Социальные группы', fontsize=12, fontweight='bold')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if 'Alive_Groups' in df.columns:
        lines_1g, labels_1g = ax1_g.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_1g, labels_1 + labels_1g, loc='best')
    else:
        ax1.legend(loc='best')

    # === 2. ЭМЕРДЖЕНТНЫЕ СТРАТЕГИИ (Обучение) ===
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Avg_Propensity_C' in df.columns:
        ax2.plot(df['Step'], df['Avg_Propensity_C'], label='Склонность к C', color='green', linewidth=1.5)
        ax2.plot(df['Step'], df['Avg_Propensity_D'], label='Склонность к D', color='red', linewidth=1.5)
        ax2.set_ylim(0, 1.05)
    ax2.set_title('🧠 Эмерджентные стратегии (Результат обучения)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')

    # === 3. ТИПЫ ИМИТАЦИИ ===
    ax3 = fig.add_subplot(gs[1, 0])
    for t in IMITATION_TYPES:
        col = f"ImitFreq_{t}"
        if col in df.columns:
            ax3.plot(df['Step'], df[col], label=IMITATION_LABELS.get(t, t), color=IMITATION_COLORS.get(t, "gray"), linewidth=1.5)
    ax3.set_title('🧬 Эволюция типов социального обучения', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')

    # === 4. РЕСУРСЫ ПО ТИПАМ ИМИТАТОРОВ ===
    ax4 = fig.add_subplot(gs[1, 1])
    for t in IMITATION_TYPES:
        col = f"ImitAvgResource_{t}"
        if col in df.columns:
            smoothed = df[col].rolling(window=max(1, len(df)//50), min_periods=1).mean()
            ax4.plot(df['Step'], smoothed, label=IMITATION_LABELS.get(t, t), color=IMITATION_COLORS.get(t, "gray"), linewidth=1.5)
    ax4.set_title('💰 Средний ресурс по типам имитаторов', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')

    # === 5. СРЕДА (Sugar + Spice + Pollution) ===
    ax5 = fig.add_subplot(gs[2, 0])
    if 'Total_Env_Sugar' in df.columns:
        ax5.plot(df['Step'], df['Total_Env_Sugar'], color='orange', linewidth=1.2, label='Sugar')
    if 'Total_Env_Spice' in df.columns:
        ax5.plot(df['Step'], df['Total_Env_Spice'], color='purple', linewidth=1.2, label='Spice')
    if 'Total_Pollution' in df.columns and df['Total_Pollution'].max() > 0:
        ax5_p = ax5.twinx()
        ax5_p.plot(df['Step'], df['Total_Pollution'], color='black', linestyle='--', linewidth=1.5, label='Загрязнение')
        ax5_p.set_ylabel('Pollution', color='black')
    ax5.set_title('Суммарный ресурс среды и Загрязнение', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')

    # === 6. ГЕНЫ ПСИХИКИ (Нейропластичность) ===
    ax6 = fig.add_subplot(gs[2, 1])
    if 'Avg_Learning_Rate' in df.columns:
        ax6.plot(df['Step'], df['Avg_Learning_Rate'], color='purple', linewidth=1.5, label='Learning Rate')
    if 'Avg_Exploration_Rate' in df.columns:
        ax6b = ax6.twinx()
        ax6b.plot(df['Step'], df['Avg_Exploration_Rate'], color='brown', linestyle='--', linewidth=1.2, label='Exploration Rate')
        ax6b.set_ylabel('Exploration Rate', color='brown')
    ax6.set_ylabel('Learning Rate', color='purple')
    ax6.set_title('🧠 Эволюция генов психики', fontsize=12, fontweight='bold')
    lines_6, labels_6 = ax6.get_legend_handles_labels()
    if 'Avg_Exploration_Rate' in df.columns:
        lines_6b, labels_6b = ax6b.get_legend_handles_labels()
        ax6.legend(lines_6 + lines_6b, labels_6 + labels_6b, loc='best')
    else:
        ax6.legend(loc='best')

    # === 7. ВРОЖДЕННЫЕ ИНСТИНКТЫ (Восстановлено) ===
    ax7 = fig.add_subplot(gs[3, 0])
    strategies = ["AlwaysC", "AlwaysD", "TFT", "WSLS", "GTFT"]
    strat_colors = {"AlwaysC": "green", "AlwaysD": "red", "TFT": "blue", "WSLS": "orange", "GTFT": "purple"}
    for s in strategies:
        col = f"Freq_{s}"
        if col in df.columns:
            ax7.plot(df['Step'], df[col], label=f'Gen: {s}', color=strat_colors.get(s, "gray"), linewidth=1.5)
    ax7.set_title('🧬 Врожденные инстинкты (Доля генов)', fontsize=12, fontweight='bold')
    ax7.legend(loc='best')

    # === 8. ГРУППОВАЯ КОНКУРЕНЦИЯ (Выделено отдельно) ===
    ax8 = fig.add_subplot(gs[3, 1])
    if 'Group_Fitness_Variance' in df.columns and df['Group_Fitness_Variance'].max() > 0:
        smoothed_var = df['Group_Fitness_Variance'].rolling(window=max(1, len(df)//50), min_periods=1).mean()
        ax8.plot(df['Step'], smoothed_var, color='darkred', linewidth=1.5, label='Fitness Variance')
        ax8.set_title('⚔️ Межгрупповая конкуренция', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Вариативность фитнеса')
        ax8.legend(loc='best')
    else:
        # Если группы выключены, показываем параметры имитации как фолбэк
        if 'Avg_Imitation_Intensity' in df.columns:
            ax8.plot(df['Step'], df['Avg_Imitation_Intensity'], color='purple', linewidth=1.5, label='Imitation Intensity')
        if 'Avg_Imitation_Rate' in df.columns:
            ax8b = ax8.twinx()
            ax8b.plot(df['Step'], df['Avg_Imitation_Rate'], color='brown', linestyle='--', linewidth=1.2, label='Imitation Rate')
            ax8b.set_ylabel('Rate', color='brown')
        ax8.set_ylabel('Intensity', color='purple')
        ax8.set_title('⚙️ Параметры имитации (Группы выкл.)', fontsize=12, fontweight='bold')
        lines_8, labels_8 = ax8.get_legend_handles_labels()
        if 'Avg_Imitation_Rate' in df.columns:
            lines_8b, labels_8b = ax8b.get_legend_handles_labels()
            ax8.legend(lines_8 + lines_8b, labels_8 + labels_8b, loc='best')
        else:
            ax8.legend(loc='best')

    # === 9. КУЛЬТУРНЫЕ ГРУППЫ (Tag-flipping) ===
    ax9 = fig.add_subplot(gs[4, 0])
    if 'Freq_Red' in df.columns and 'Freq_Blue' in df.columns:
        ax9.plot(df['Step'], df['Freq_Red'], label='Red (Культурная группа)', color='red', linewidth=1.5)
        ax9.plot(df['Step'], df['Freq_Blue'], label='Blue (Культурная группа)', color='blue', linewidth=1.5)
        ax9.set_ylim(0, 1.05)
    ax9.set_title('🏷️ Культурная динамика (Tag-flipping)', fontsize=12, fontweight='bold')
    ax9.legend(loc='best')

    # === 10. РАЗНООБРАЗИЕ ТЕГОВ ===
    ax10 = fig.add_subplot(gs[4, 1])
    if 'Avg_Tag_Diversity' in df.columns:
        ax10.plot(df['Step'], df['Avg_Tag_Diversity'], color='darkorange', linewidth=1.5, label='Средняя доля "1" в тегах')
        ax10.set_ylim(0, 1.05)
    ax10.set_title('🧬 Культурное разнообразие (Доля активных тегов)', fontsize=12, fontweight='bold')
    ax10.legend(loc='best')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Графики сохранены: {save_path}")
    else:
        plt.show()


def plot_imitation_comparison(df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    types, freqs, colors = [], [], []
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
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{freq:.1%}', va='center', fontsize=9)

    ax2 = axes[1]
    resources = []
    for t in IMITATION_TYPES:
        col = f"ImitAvgResource_{t}"
        if col in df.columns:
            quarter = len(df) // 4
            resources.append(df[col].iloc[-quarter:].mean())
        else:
            resources.append(0)
            
    bars2 = ax2.barh(types, resources, color=colors)
    ax2.set_xlabel('Средний ресурс (последняя четверть)')
    ax2.set_title('💪 Средняя приспособленность', fontweight='bold')
    for bar, res in zip(bars2, resources):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{res:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сравнительный график сохранён: {save_path}")
    else:
        plt.show()
