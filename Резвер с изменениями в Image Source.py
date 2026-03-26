import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

def get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq):
    virtual_sources = []
    for n in range(-max_order, max_order + 1):
        for m in range(-max_order, max_order + 1):
            order = abs(n) + abs(m)
            if order == 0 or order > max_order:
                continue
            
            left_count = abs(n) if n < 0 else 0
            right_count = n if n > 0 else 0
            bottom_count = abs(m) if m < 0 else 0
            top_count = m if m > 0 else 0
            
            effective_energy_factors = []
            for freq_idx in range(6):
                factor = (
                    (reflection_coeffs_freq['left'][freq_idx] ** left_count) *
                    (reflection_coeffs_freq['right'][freq_idx] ** right_count) *
                    (reflection_coeffs_freq['bottom'][freq_idx] ** bottom_count) *
                    (reflection_coeffs_freq['top'][freq_idx] ** top_count)
                )
                effective_energy_factors.append(factor)
            
            if all(f == 0 for f in effective_energy_factors):
                continue
            
            x = n * room_width + (source[0] if n % 2 == 0 else room_width - source[0])
            y = m * room_height + (source[1] if m % 2 == 0 else room_height - source[1])
            virtual_sources.append({
                'pos': [x, y], 
                'order': order,
                'energy_factors': effective_energy_factors,
                'left_refl': left_count,
                'right_refl': right_count,
                'top_refl': top_count,
                'bottom_refl': bottom_count
            })
    return virtual_sources

def find_wall_intersections(p1, p2, room_width, room_height):
    intersections = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx != 0:
        t = (0 - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((0, y, 'left', t))
        t = (room_width - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((room_width, y, 'right', t))

    if dy != 0:
        t = (0 - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, 0, 'bottom', t))
        t = (room_height - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, room_height, 'top', t))

    return sorted(intersections, key=lambda inter: inter[3])

def plot_reflection_path(ax, points, color):
    xs, ys = zip(*points)
    ax.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.7)
    for p in points[1:-1]:
        ax.plot(p[0], p[1], 'bo', markersize=6, alpha=0.7)

def calculate_rt60(times, energies, freq_label):
    if len(energies) < 2:
        return None
    
    energies = np.array(energies)
    mask = energies > 0
    valid_times = np.array(times)[mask]
    valid_energies = energies[mask]
    
    if len(valid_energies) < 2:
        return None
    
    levels_db = 10 * np.log10(valid_energies / np.max(valid_energies))
    sorted_indices = np.argsort(valid_times)
    sorted_times = valid_times[sorted_indices]
    sorted_levels = levels_db[sorted_indices]
    
    if sorted_levels[0] - sorted_levels[-1] < 5:
        return None
    
    start_idx = 0
    end_idx = len(sorted_times) - 1
    
    for i in range(len(sorted_levels)):
        if sorted_levels[i] <= -5:
            start_idx = i
            break
    
    for i in range(len(sorted_levels)-1, -1, -1):
        if sorted_levels[i] >= -25:
            end_idx = i
            break
    
    if end_idx <= start_idx:
        if len(sorted_times) >= 3:
            selected_times = sorted_times
            selected_levels = sorted_levels
        else:
            return None
    else:
        selected_times = sorted_times[start_idx:end_idx+1]
        selected_levels = sorted_levels[start_idx:end_idx+1]
    
    if len(selected_times) < 2:
        return None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(selected_times, selected_levels)
    
    if slope >= 0:
        return None
    
    rt60 = -60.0 / slope if slope != 0 else float('inf')
    
    return {
        'rt60': rt60,
        'slope': slope,
        'r_squared': r_value**2,
        'times_used': selected_times,
        'levels_used': selected_levels
    }

def draw_room_and_sources(room_width, room_height, source, receiver, max_order, reflection_coeffs_freq):
    fig, ax = plt.subplots(figsize=(12, 10))
    room_patch = patches.Rectangle((0, 0), room_width, room_height, linewidth=2,
                                  edgecolor='orange', facecolor='none', linestyle='-')
    ax.add_patch(room_patch)
    ax.plot(source[0], source[1], 'ro', markersize=10, label='Источник')
    ax.plot(receiver[0], receiver[1], 'go', markersize=10, label='Приёмник')
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]],
            color='purple', linestyle='-', linewidth=2, alpha=0.8, label='Прямой путь')

    speed_of_sound = 343

    # Прямой путь
    direct_dist = np.linalg.norm(np.array(source) - np.array(receiver))
    direct_time = direct_dist / speed_of_sound
    direct_energy = 1 / direct_dist ** 2
    print(f"Прямой путь: расстояние = {direct_dist:.2f} м, время = {direct_time:.4f} с, энергия = {direct_energy:.5f}")

    virtual_sources = get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq)

    freq_labels = ['125 Гц', '250 Гц', '500 Гц', '1 кГц', '2 кГц', '4 кГц']
    times_freq = [[] for _ in range(6)]
    energies_freq = [[] for _ in range(6)]
    distances = []

    # === НОВОЕ: Параметры для импульсной характеристики и Фурье ===
    fs = 48000              # частота дискретизации, Гц
    ir_duration = 1.0       # длительность ИХ, с
    N = int(fs * ir_duration)
    h = np.zeros(N)         # импульсная характеристика
    freq_idx_fft = 2        # используем полосу 500 Гц (индекс 2)

    # === Добавить прямой путь в ИХ ===
    n_direct = int(round(direct_time * fs))
    if 0 <= n_direct < N:
        h[n_direct] += 1.0 / direct_dist  # амплитуда прямого пути

    for vs in virtual_sources:
        pos = vs['pos']
        order = vs['order']
        energy_factors = vs['energy_factors']
        reflections = find_wall_intersections(pos, receiver, room_width, room_height)
        path_points = [pos] + [(ix[0], ix[1]) for ix in reflections] + [receiver]
        r = 0
        for i in range(len(path_points) - 1):
            seg_length = np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i]))
            r += seg_length
        distances.append(r)
        print(f"Луч порядка {order}: длина = {r:.2f} м")
        
        # === НОВОЕ: Вклад этого луча в импульсную характеристику ===
        t = r / speed_of_sound
        n = int(round(t * fs))
        if 0 <= n < N:
            # амплитуда = 1/r * sqrt(энергия отражений) как в MATLAB
            amp = (1.0 / r) * np.sqrt(energy_factors[freq_idx_fft])
            h[n] += amp
        
        # Старый код для RT60
        for freq_idx in range(6):
            Amplituda = (1/r)*np.sqrt(energy_factors[freq_idx])
            time_sec = r / speed_of_sound
            if Amplituda > 0:
                times_freq[freq_idx].append(time_sec)
                energies_freq[freq_idx].append(Amplituda)
                print(f"  {freq_labels[freq_idx]}: время = {time_sec:.4f} с, энергия = {Amplituda:.5f}")

        plot_reflection_path(ax, path_points, color='y')
        ax.plot(pos[0], pos[1], 'ro', markersize=5, alpha=0.7)
        offset_x = (pos[0] // room_width) * room_width
        offset_y = (pos[1] // room_height) * room_height
        virtual_room = patches.Rectangle(
            (offset_x, offset_y), room_width, room_height,
            linewidth=1, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.5
        )
        ax.add_patch(virtual_room)

    ax.set_xlim(-room_width, 2 * room_width)
    ax.set_ylim(-room_height, 2 * room_height)
    plt.gca().tick_params(labelsize=20)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=17)
    plt.show()

    # === НОВОЕ: График импульсной характеристики ===
    time_axis = np.arange(N) / fs
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, h)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title(f'Импульсная характеристика комнаты (полоса {freq_labels[freq_idx_fft]})')
    plt.grid(True, alpha=0.3)
    
    # === НОВОЕ: Фурье-преобразование ===
    H = np.fft.rfft(h)
    freqs_fft = np.fft.rfftfreq(N, d=1/fs)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(freqs_fft, np.abs(H))
    plt.xlabel('Частота (Гц)')
    plt.ylabel('|H(f)|')
    plt.title('Частотная характеристика комнаты')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Старый код RT60 (остаётся без изменений)
    rt60_results = {}
    for freq_idx in range(6):
        if len(energies_freq[freq_idx]) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(times_freq[freq_idx], energies_freq[freq_idx], 
                       c='green', alpha=0.7, label=f'Точки: E(r) для {freq_labels[freq_idx]}')

            def inv_r_squared(r, C):
                return C / (r ** 2)

            try:
                popt, pcov = curve_fit(inv_r_squared, times_freq[freq_idx], 
                                     energies_freq[freq_idx], p0=(1), maxfev=5000)
                C_fit = popt[0]
                print(f"\nПараметр аппроксимации для {freq_labels[freq_idx]} E(r) = C / r^2: C = {C_fit:.4f}")
                r_smooth = np.linspace(min(times_freq[freq_idx]), 
                                     max(times_freq[freq_idx]), 500)
                e_smooth = inv_r_squared(r_smooth, C_fit)
                plt.plot(r_smooth, e_smooth, 'b-', linewidth=2)
            except RuntimeError:
                print(f"\nНе удалось подобрать параметры для {freq_labels[freq_idx]} E(r) = C / r^2.")
            
            plt.title(f'Зависимость энергии отражённых лучей от времени\n'
                     f'для виртуальных источников до {max_order} порядка ({freq_labels[freq_idx]})')
            plt.xlabel('Время, с')
            plt.ylabel('Энергия')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.show()
        
        if len(times_freq[freq_idx]) > 1:
            energies_db = 10 * np.log10(np.array(energies_freq[freq_idx]) / np.max(energies_freq[freq_idx]))
            
            plt.figure(figsize=(10, 6))
            plt.scatter(times_freq[freq_idx], energies_db, c='red', alpha=0.7, 
                       label=f'Уровни звука по времени для {freq_labels[freq_idx]}')
            
            rt60_result = calculate_rt60(times_freq[freq_idx], energies_freq[freq_idx], freq_labels[freq_idx])
            if rt60_result:
                rt60_results[freq_labels[freq_idx]] = rt60_result['rt60']
                
                times_for_line = rt60_result['times_used']
                levels_for_line = rt60_result['levels_used']
                slope = rt60_result['slope']
                extended_time = np.linspace(times_for_line[0], times_for_line[0] + rt60_result['rt60'], 100)
                extended_level = slope * (extended_time - times_for_line[0]) + levels_for_line[0]
                
                plt.plot(extended_time, extended_level, 'b--', linewidth=2, 
                        label=f'Линия регрессии (RT60 ≈ {rt60_result["rt60"]:.3f} с)')
                
                print(f"\nRT60 для {freq_labels[freq_idx]}: {rt60_result['rt60']:.3f} секунд")
                print(f"Склон затухания: {rt60_result['slope']:.3f} дБ/сек")
                print(f"Коэффициент детерминации R²: {rt60_result['r_squared']:.3f}")
            else:
                print(f"\nНе удалось рассчитать RT60 для {freq_labels[freq_idx]}")
            
            plt.title(f'Кривая затухания уровня звука во времени\n({freq_labels[freq_idx]})')
            plt.xlabel('Время (с)')
            plt.ylabel('Уровень (относительный, дБ)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.show()
    
    print("ТАБЛИЦА RT60 ДЛЯ ВСЕХ ЧАСТОТНЫХ ДИАПАЗОНОВ")
    print("Частота\t\tRT60 (с)")
    print("-" * 25)
    for freq_label in freq_labels:
        if freq_label in rt60_results:
            print(f"{freq_label:<12}\t{rt60_results[freq_label]:.3f}")
        else:
            print(f"{freq_label:<12}\tНедоступно")

# === Параметры ===
room_width = 10
room_height = 8
source = [2, 3]
receiver = [5, 5]
max_reflection_order = 14
left_freq = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
right_freq = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
top_freq = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bottom_freq = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
reflection_coeffs_freq = {
    'left': left_freq,
    'right': right_freq,
    'top': top_freq,
    'bottom': bottom_freq
}

# Расчёт RT60 по Сэбину
s1 = 10
s2 = 8
V = s1 * s2
for i in np.arange(0.1, 0.7, 0.1):
    A = s1*i + s2*i + s1*i + s2*i
    rt60 = (0.161*V) / A
    print(f'{rt60:.3f} - Время реверберации для {i:.1f} поглощения (отражение {1-i:.1f})')

draw_room_and_sources(room_width, room_height, source, receiver, max_reflection_order, reflection_coeffs_freq)
