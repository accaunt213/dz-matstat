import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
#a
n = 25
sample = np.random.exponential(scale=1, size=n)

median = np.median(sample)

data_range = np.max(sample) - np.min(sample)

mean_x = np.mean(sample)
m2 = np.mean((sample - mean_x)**2)
m3 = np.mean((sample - mean_x)**3)
skew_manual = m3 / m2**1.5

unique, counts = np.unique(sample, return_counts=True)
max_count = np.max(counts)
if max_count == 1:
    mode = "моды нет"
else:
    mode = unique[np.argmax(counts)]

print("Мода:", mode)
print("Выборка:",sample)
print("Медиана",median)
print("Размах:", data_range)
print("Коэффициент асимметрии:", skew_manual)

#b
sample_sorted = np.sort(sample)
y = np.arange(1, len(sample_sorted)+1) / len(sample_sorted)

plt.figure()
plt.step(sample_sorted, y, where='post')
plt.xlabel('x')
plt.title('Эмпирическая функция распределения')
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('ecdf.png')

K = int(np.ceil(1 + np.log2(n)))

x_min = np.min(sample)
x_max = np.max(sample)
delta = (x_max - x_min) / K

bins = np.linspace(x_min, x_max, K+1)

counts, _ = np.histogram(sample, bins=bins)
rel_freqs = counts / n

plt.figure()
plt.bar(bins[:-1], rel_freqs, width=delta, align='edge', alpha=0.6, color='skyblue', edgecolor='black')
x_theor = np.linspace(0, x_max, 200)
plt.plot(x_theor, np.exp(-x_theor), 'r-', linewidth=2, label='Теоретическая плотность $e^{-x}$')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Гистограмма')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('histogram.png')

plt.figure()
plt.boxplot(sample, vert=False, patch_artist=True)
plt.xlabel('Значения')
plt.ylabel('Выборка')
plt.title('Boxplot')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('boxplot.png')

#c
x_bar = np.mean(sample)
s2 = np.var(sample, ddof=1)
se = np.sqrt(s2 / n)

t = np.linspace(x_bar - 4*se, x_bar + 4*se, 200)

f_clt = norm.pdf(t, loc=x_bar, scale=se)

N = 1000
boot_means = []
for _ in range(N):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    boot_means.append(np.mean(boot_sample))

plt.figure(figsize=(8, 5))
plt.hist(boot_means, bins=30, density=True, alpha=0.6, color='skyblue',
         edgecolor='black', label='Bootstrap')
plt.plot(t, f_clt, 'r-', linewidth=2, label='ЦПТ')
plt.xlabel('Среднее арифметическое')
plt.ylabel('Плотность')
plt.title('Сравнение Bootstrap среднего с нормальным приближением')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('compare_mean_hist.png')

#d
def skewness_manual(x):
    mean_x = np.mean(x)
    m2 = np.mean((x - mean_x)**2)
    m3 = np.mean((x - mean_x)**3)
    return m3 / m2**1.5
orig_skew = skewness_manual(sample)

N = 1000
n = len(sample)
boot_skews = []
for _ in range(N):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    boot_skews.append(skewness_manual(boot_sample))

kde = gaussian_kde(boot_skews)
x_range = np.linspace(min(boot_skews), max(boot_skews), 200)
density = kde(x_range)

prob_less_than_1 = np.mean(np.array(boot_skews) < 1)
print("Оценка вероятности P(коэффициент асимметрии < 1):",prob_less_than_1)

plt.figure(figsize=(8, 5))
plt.hist(boot_skews, bins=30, density=True, alpha=0.6, color='lightgreen',edgecolor='black', label='Bootstrap-значения')
plt.plot(x_range, density, 'b-', linewidth=2, label='KDE (Bootstrap)')
plt.axvline(x=orig_skew, color='purple', linestyle=':', linewidth=2,label=f'Исходный g1 = {orig_skew:.2f}')
plt.xlabel('Коэффициент асимметрии')
plt.ylabel('Плотность')
plt.title('Бутстраповская оценка плотности распределения коэффициента асимметрии')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('skewness_bootstrap.png')

#e
M = 1000   
N = 1000

true_medians = []
for _ in range(M):
    sim_sample = np.random.exponential(scale=1, size=n)
    true_medians.append(np.median(sim_sample))

boot_medians = []
for _ in range(N):
    boot_sample = np.random.choice(sample, size=n, replace=True)
    boot_medians.append(np.median(boot_sample))

plt.figure(figsize=(8, 5))
plt.hist(true_medians, bins=30, density=True, alpha=0.5, color='blue',edgecolor='black', label='Истинное распределение')
plt.hist(boot_medians, bins=30, density=True, alpha=0.5, color='red',edgecolor='black', label='Бутстраповская оценка')
plt.xlabel('Медиана')
plt.ylabel('Плотность')
plt.title('Сравнение распределений выборочной медианы')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('median_comparison_hist.png')