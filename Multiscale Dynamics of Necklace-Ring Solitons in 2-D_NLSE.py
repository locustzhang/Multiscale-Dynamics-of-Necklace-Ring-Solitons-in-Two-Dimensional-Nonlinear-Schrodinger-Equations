import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks
from tqdm import tqdm
import os
import random
import time
from dataclasses import dataclass, fields
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. CINEMATIC AESTHETICS =====================

# 主题设置 - 极致惊艳的暗黑霓虹主题
DARK_MODE = True  # True = 暗黑霓虹主题（极致视觉效果）

if DARK_MODE:
    BG_COLOR = '#000000'  # 纯黑背景 (OLED 完美显示)
    FG_COLOR = 'white'
    GRID_COLOR = '#1a1a2e'
else:
    BG_COLOR = 'white'
    FG_COLOR = 'black'
    GRID_COLOR = '#CCCCCC'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': FG_COLOR,
    'text.color': FG_COLOR,
    'xtick.color': FG_COLOR,
    'ytick.color': FG_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.linestyle': '-',
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stixsans',
})


def style_axis(ax, spines_off=['top', 'right']):
    for s in spines_off:
        ax.spines[s].set_visible(False)
    ax.tick_params(direction='out', width=1.0, length=4)


# 全局种子
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "Necklace_Soliton_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 2. 复数域着色 =====================

def complex_to_rgb(Z, max_val=None, gamma=0.5):
    """将复数场转换为RGB：色相=相位，亮度=振幅"""
    abs_Z = np.abs(Z)
    angle_Z = np.angle(Z)

    if max_val is None:
        max_val = np.percentile(abs_Z, 99.5)

    v = np.clip(abs_Z / max_val, 0, 1) ** gamma
    h = (angle_Z + np.pi) / (2 * np.pi)
    s = np.ones_like(v)
    s[v < 0.05] = v[v < 0.05] * 20

    hsv = np.dstack((h, s, v))
    return mcolors.hsv_to_rgb(hsv)


# ===================== 3. 高精度物理引擎 =====================

@dataclass
class SimConfig2D:
    """仿真配置 - 优化参数确保稳定性"""
    Nx: int = 512
    Ny: int = 512
    Lx: float = 30.0
    Ly: float = 30.0
    dt: float = 0.00005  # 更小时间步长
    T: float = 8.0

    # 项链参数
    N_peaks: int = 6
    R_ring: float = 8.0
    A_peak: float = 0.8  # 更低振幅
    w_peak: float = 1.5

    # 物理参数 - 进一步降低确保稳定
    g: float = -0.1  # 极弱非线性
    Omega: float = 0.05  # 极弱旋转

    seed: int = 42

    def __post_init__(self):
        self.dx = 2 * self.Lx / self.Nx
        self.dy = 2 * self.Ly / self.Ny
        self.x = torch.linspace(-self.Lx, self.Lx, self.Nx, device=DEVICE)
        self.y = torch.linspace(-self.Ly, self.Ly, self.Ny, device=DEVICE)


class NecklaceNLSE2D:
    """2D NLSE 求解器"""

    def __init__(self, cfg: SimConfig2D):
        self.cfg = cfg
        self.X, self.Y = torch.meshgrid(cfg.x, cfg.y, indexing='ij')

        kx = (2 * np.pi / (2 * cfg.Lx)) * torch.fft.fftfreq(cfg.Nx, d=1).to(DEVICE) * cfg.Nx
        ky = (2 * np.pi / (2 * cfg.Ly)) * torch.fft.fftfreq(cfg.Ny, d=1).to(DEVICE) * cfg.Ny
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.prop_half = torch.exp(-0.5j * 0.5 * self.K2 * cfg.dt)

    def necklace_initial(self, velocity=0.0, direction='inward'):
        """生成项链环初始态"""
        cfg = self.cfg
        psi = torch.zeros(cfg.Nx, cfg.Ny, dtype=torch.complex128, device=DEVICE)
        theta_peaks = torch.linspace(0, 2 * np.pi, cfg.N_peaks + 1, device=DEVICE)[:-1]

        for theta in theta_peaks:
            x0 = cfg.R_ring * torch.cos(theta)
            y0 = cfg.R_ring * torch.sin(theta)
            r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            gaussian = torch.exp(-r_sq / cfg.w_peak ** 2)
            r_local = torch.sqrt(r_sq)
            phase = -velocity * r_local if direction == 'inward' else velocity * r_local
            psi += cfg.A_peak * gaussian * torch.exp(1j * phase)

        if cfg.Omega > 0:
            psi *= torch.exp(1j * cfg.Omega * torch.atan2(self.Y, self.X))
        return psi

    def two_ring_collision(self, R1=6.0, R2=12.0, v=2.0):
        """双环碰撞初始态"""
        cfg = self.cfg
        psi_inner = torch.zeros(cfg.Nx, cfg.Ny, dtype=torch.complex128, device=DEVICE)
        psi_outer = torch.zeros(cfg.Nx, cfg.Ny, dtype=torch.complex128, device=DEVICE)
        theta_peaks = torch.linspace(0, 2 * np.pi, cfg.N_peaks + 1, device=DEVICE)[:-1]

        for theta in theta_peaks:
            # 内环外扩
            x0, y0 = R1 * torch.cos(theta), R1 * torch.sin(theta)
            r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            gaussian = torch.exp(-r_sq / cfg.w_peak ** 2)
            r_local = torch.sqrt(r_sq)
            psi_inner += cfg.A_peak * gaussian * torch.exp(1j * v * r_local)

            # 外环内收
            x0, y0 = R2 * torch.cos(theta), R2 * torch.sin(theta)
            r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            gaussian = torch.exp(-r_sq / cfg.w_peak ** 2)
            r_local = torch.sqrt(r_sq)
            psi_outer += cfg.A_peak * gaussian * torch.exp(1j * (-v) * r_local)

        return psi_inner + psi_outer

    def energy(self, psi):
        """计算总能量"""
        psi_k = torch.fft.fft2(psi)
        E_kin = 0.5 * torch.sum(self.K2 * torch.abs(psi_k) ** 2) / (
                    self.cfg.Nx * self.cfg.Ny) * self.cfg.dx * self.cfg.dy
        rho = torch.abs(psi) ** 2
        E_pot = 0.5 * self.cfg.g * torch.sum(rho ** 2) * self.cfg.dx * self.cfg.dy
        return (E_kin + E_pot).real.item()

    def angular_momentum(self, psi):
        """计算角动量"""
        psi_k = torch.fft.fft2(psi)
        dpsi_dx = torch.fft.ifft2(1j * self.KX * psi_k)
        dpsi_dy = torch.fft.ifft2(1j * self.KY * psi_k)
        Lz = torch.imag(torch.sum(torch.conj(psi) * (self.X * dpsi_dy - self.Y * dpsi_dx))) * self.cfg.dx * self.cfg.dy
        return Lz.item()

    def radial_profile(self, rho, n_bins=100):
        """径向分布"""
        R = torch.sqrt(self.X ** 2 + self.Y ** 2).cpu().numpy()
        rho_np = rho.cpu().numpy() if torch.is_tensor(rho) else rho
        r_bins = np.linspace(0, self.cfg.Lx, n_bins)
        rho_radial, r_centers = [], []
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            if np.sum(mask) > 0:
                rho_radial.append(np.mean(rho_np[mask]))
                r_centers.append((r_bins[i] + r_bins[i + 1]) / 2)
        return np.array(r_centers), np.array(rho_radial)

    def run_simulation(self, psi0, save_full=True):
        """运行模拟 - 优化内存，保存约200帧"""
        cfg = self.cfg
        steps = int(cfg.T / cfg.dt)
        save_interval = max(1, steps // 200)  # 确保只保存约200帧，节省内存

        psi = psi0.clone()
        E0, Lz0 = self.energy(psi), self.angular_momentum(psi)

        data = {'t': [], 'E': [], 'Lz': [], 'E_err': [], 'Lz_err': [],
                'rho_max': [], 'field': [] if save_full else None}

        # 静默模式检测：批量扫描时禁用进度条
        silent_mode = not save_full and steps > 10000

        if silent_mode:
            # 批量扫描模式：极简输出，每25%报告一次
            print(f"    [Run] steps={steps}, dt={cfg.dt}, grid={cfg.Nx}x{cfg.Ny}")
            milestones = [steps // 4, steps // 2, 3 * steps // 4, steps - 1]
            milestone_idx = 0
        else:
            pbar = tqdm(range(steps), desc="Simulating", colour='cyan',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for i in range(steps):
            # Strang splitting
            psi = torch.fft.ifft2(torch.fft.fft2(psi) * self.prop_half)
            rho = torch.abs(psi) ** 2
            psi = psi * torch.exp(1j * cfg.g * rho * cfg.dt)
            psi = torch.fft.ifft2(torch.fft.fft2(psi) * self.prop_half)

            if i % save_interval == 0:
                t = i * cfg.dt
                E, Lz = self.energy(psi), self.angular_momentum(psi)
                rho_cpu = torch.abs(psi) ** 2

                data['t'].append(t)
                data['E'].append(E)
                data['Lz'].append(Lz)
                data['E_err'].append(abs(E - E0) / abs(E0))
                data['Lz_err'].append(abs(Lz - Lz0))  # 绝对误差，避免除以零问题
                data['rho_max'].append(torch.max(rho_cpu).item())

                if save_full:
                    data['field'].append(psi.cpu().numpy())

                if silent_mode:
                    if milestone_idx < len(milestones) and i >= milestones[milestone_idx]:
                        pct = 25 * (milestone_idx + 1)
                        print(
                            f"      {pct}%: E={E:.3f}, err={data['E_err'][-1]:.2e}, rho_max={data['rho_max'][-1]:.3f}")
                        milestone_idx += 1
                else:
                    pbar.set_postfix(E=f"{E:.2f}", err=f"{data['E_err'][-1]:.2e}")
                    pbar.update(save_interval)

        if not silent_mode:
            pbar.close()

        for k in ['t', 'E', 'Lz', 'E_err', 'Lz_err', 'rho_max']:
            data[k] = np.array(data[k])
        if save_full:
            data['field'] = np.array(data['field'])

        data['E0'], data['Lz0'] = E0, Lz0
        data['max_E_err'] = np.max(data['E_err'])
        data['max_Lz_err'] = np.max(data['Lz_err'])

        return data


# ===================== 4. 可视化 =====================

def create_fig1_domain(data, sys, cfg):
    """Figure 1: 复数域着色 + 霓虹光晕效果"""
    from scipy.ndimage import gaussian_filter

    field, t = data['field'], data['t']
    indices = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    labels = ['Initial', 'Formation', 'Rotation', 'Steady']

    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.3)

    max_amp = np.max(np.abs(field))

    for idx, (i, label) in enumerate(zip(indices, labels)):
        ax = fig.add_subplot(gs[0, idx])

        # 提取振幅
        amp = np.abs(field[i])

        # 【光晕效果】高斯模糊叠加
        glow = gaussian_filter(amp, sigma=3.0)
        glow_norm = glow / np.max(glow)

        # 绘制光晕层（青色霓虹）
        ax.imshow(glow_norm, extent=[-cfg.Lx, cfg.Lx, -cfg.Ly, cfg.Ly],
                  cmap='Blues', alpha=0.4, origin='lower', vmin=0.3, vmax=1.0)

        # 主图像：复数域着色
        rgb = complex_to_rgb(field[i], max_val=max_amp, gamma=0.5)
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower',
                  extent=[-cfg.Lx, cfg.Lx, -cfg.Ly, cfg.Ly],
                  interpolation='bilinear')

        # 装饰
        ax.add_artist(plt.Circle((0, 0), cfg.R_ring, color='cyan', fill=False, ls='--', alpha=0.6, lw=1))
        ax.plot(0, 0, '+', color='#FF00AA', markersize=10, mew=2)  # 粉色中心标记

        ax.set_title(f"{label}\n($t={t[i]:.2f}$)", fontsize=10, fontweight='bold', color='white')
        ax.set_xlabel('$x$', color='white')
        if idx == 0: ax.set_ylabel('$y$', color='white')
        style_axis(ax)
        ax.set_aspect('equal')
        ax.tick_params(colors='white')

    # 相位色轮
    ax_wheel = fig.add_subplot(gs[0, 4])
    theta = np.linspace(0, 2 * np.pi, 256)
    r = np.linspace(0, 1, 128)
    T, R = np.meshgrid(theta, r)
    HSV = np.dstack((T / (2 * np.pi), np.ones_like(T), R))
    RGB = mcolors.hsv_to_rgb(HSV)
    Xc, Yc = R * np.cos(T), R * np.sin(T)
    RGB[Xc ** 2 + Yc ** 2 > 1] = [0, 0, 0]  # 纯黑背景
    ax_wheel.imshow(RGB, extent=[-1, 1, -1, 1], origin='lower')
    ax_wheel.set_title('Phase', fontsize=9, color='white')
    ax_wheel.axis('off')

    plt.suptitle(f"NEON NECKLACE SOLITON | N={cfg.N_peaks} | g={cfg.g} | Omega={cfg.Omega}",
                 fontsize=12, fontweight='bold', y=1.02, color='#00F3FF')
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Domain.png", bbox_inches='tight', dpi=600, facecolor='black')
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Domain.pdf", bbox_inches='tight', facecolor='black')
    plt.close()


def create_fig2_collision(data, sys, cfg):
    """Figure 2: 碰撞序列 + 增强清晰度"""
    from scipy.ndimage import gaussian_filter

    field, t = data['field'], data['t']

    # 找碰撞时刻
    ring_sep = []
    for i in range(len(t)):
        r_c, rho_r = sys.radial_profile(np.abs(field[i]) ** 2)
        peaks, _ = find_peaks(rho_r, height=np.max(rho_r) * 0.1, distance=5)
        ring_sep.append(abs(r_c[peaks[0]] - r_c[peaks[1]]) if len(peaks) >= 2 else 0)
    collision_idx = np.argmin(ring_sep)

    idx_list = [max(0, collision_idx - 60), max(0, collision_idx - 40), max(0, collision_idx - 20),
                max(0, collision_idx - 8), collision_idx,
                min(len(t) - 1, collision_idx + 12), min(len(t) - 1, collision_idx + 30),
                min(len(t) - 1, collision_idx + 50), len(t) - 1]
    labels = ['(a) App.I', '(b) App.II', '(c) Near', '(d) Pre', '(e) COLLISION',
              '(f) Merg.I', '(g) Merg.II', '(h) Post', '(i) Final']

    fig = plt.figure(figsize=(15, 15), facecolor='black')
    gs = GridSpec(3, 3, wspace=0.2, hspace=0.3)

    # 全局归一化确保一致性
    global_max = np.max(np.abs(field))

    for idx, (i, label) in enumerate(zip(idx_list, labels)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor('black')

        # 提取振幅
        amp = np.abs(field[i])

        # 【增强光晕】更大sigma，更低透明度
        glow = gaussian_filter(amp, sigma=2.0)  # 更小sigma = 更锐利
        glow_norm = glow / global_max

        # 多层光晕效果
        ax.imshow(glow_norm, extent=[-cfg.Lx, cfg.Lx, -cfg.Ly, cfg.Ly],
                  cmap='hot', alpha=0.2, origin='lower', vmin=0.1, vmax=0.8)

        # 主图像：增强对比度
        rgb = complex_to_rgb(field[i], max_val=global_max, gamma=0.4)  # 更低gamma = 更高对比
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower',
                  extent=[-cfg.Lx, cfg.Lx, -cfg.Ly, cfg.Ly],
                  interpolation='bicubic')  # 更高质量插值

        # 添加等高线增强结构可见性
        ax.contour(sys.X.cpu().numpy(), sys.Y.cpu().numpy(), amp,
                   levels=[global_max * 0.3, global_max * 0.6],
                   colors=['cyan', 'yellow'], alpha=0.4, linewidths=0.5)

        # 碰撞帧高亮
        if idx == 4:
            for spine in ax.spines.values():
                spine.set_color('#FF00AA')
                spine.set_linewidth(4)
            ax.set_title(label, fontsize=11, fontweight='bold', color='#FF00AA')
        else:
            ax.set_title(label, fontsize=10, fontweight='bold', color='white')

        ax.set_xlabel('$x$', color='white', fontsize=9)
        ax.set_ylabel('$y$', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.text(0.02, 0.98, f'$t={t[i]:.2f}$', transform=ax.transAxes, fontsize=9, va='top',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan'))
        style_axis(ax)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)  # 聚焦中心区域
        ax.set_ylim(-15, 15)

    plt.suptitle("TWO-RING COLLISION | Enhanced Clarity", fontsize=16, fontweight='bold', color='#00F3FF', y=0.98)
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Collision.png", bbox_inches='tight', dpi=600, facecolor='black')
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Collision.pdf", bbox_inches='tight', facecolor='black')
    plt.close()


def create_fig3_physics(data, cfg):
    """Figure 3: 物理量"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t = data['t']

    # 能量误差
    ax = axes[0, 0]
    ax.semilogy(t, data['E_err'] * 100, 'b-', lw=2)
    ax.axhline(1, color='r', ls='--', lw=1, alpha=0.7, label='1%')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Energy Error (%)')
    ax.set_title('(a) Energy Conservation', fontweight='bold')
    ax.legend()
    style_axis(ax)
    ax.text(0.98, 0.98, f"Max: {data['max_E_err']:.2e}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 角动量误差（绝对值）
    ax = axes[0, 1]
    ax.semilogy(t, data['Lz_err'], 'm-', lw=2)
    ax.axhline(0.01, color='r', ls='--', lw=1, alpha=0.7, label='0.01 threshold')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$|L_z - L_{z0}|$ (absolute)')
    ax.set_title('(b) Angular Momentum')
    ax.legend()
    style_axis(ax)
    ax.text(0.98, 0.98, f"Max: {data['max_Lz_err']:.2e}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 峰值密度
    ax = axes[1, 0]
    ax.plot(t, data['rho_max'], 'g-', lw=2)
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\max|\psi|^2$')
    ax.set_title('(c) Peak Density', fontweight='bold')
    style_axis(ax)

    # 总能量
    ax = axes[1, 1]
    ax.plot(t, data['E'], 'purple', lw=2)
    ax.axhline(data['E0'], color='r', ls='--', lw=1, alpha=0.7, label=f'$E_0={data["E0"]:.1f}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$E$')
    ax.set_title('(d) Total Energy', fontweight='bold')
    ax.legend()
    style_axis(ax)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Physics.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Physics.pdf", bbox_inches='tight')
    plt.close()


def create_fig4_radial(data, sys, cfg):
    """Figure 4: 径向分布"""
    field, t = data['field'], data['t']
    indices = np.linspace(0, len(t) - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for idx, i in enumerate(indices):
        ax = axes[idx // 3, idx % 3]
        rho = np.abs(field[i]) ** 2
        r_c, rho_r = sys.radial_profile(rho)

        ax.fill_between(r_c, 0, rho_r, alpha=0.3, color='blue')
        ax.plot(r_c, rho_r, 'b-', lw=2)

        peaks, _ = find_peaks(rho_r, height=np.max(rho_r) * 0.1, distance=5)
        for p in peaks:
            ax.plot(r_c[p], rho_r[p], 'ro', markersize=6)

        ax.set_title(f'$t={t[i]:.2f}$ ({len(peaks)} rings)', fontsize=10, fontweight='bold')
        ax.set_xlabel('$r$')
        ax.set_ylabel(r'$\langle\rho\rangle_\theta$')
        ax.set_xlim(0, cfg.Lx)
        style_axis(ax)

    plt.suptitle("Radial Density Evolution", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Radial.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Radial.pdf", bbox_inches='tight')
    plt.close()


# ===================== 5. 主程序 =====================

def print_banner():
    banner = """
================================================================================
           NECKLACE SOLITON DYNAMICS v3.1 - ULTRA STABLE
           Minimal Nonlinearity | Maximum Precision | Neon Glow
================================================================================
    """
    print(banner)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {DEVICE} | Seed: {SEED}")
    print(f"Grid: 512x512 | dt: 0.00005 | g: -0.1 | A: 0.8 | Omega: 0.05")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 80)


def print_results(data, comp_time, name):
    print(f"\n{'-' * 80}")
    print(f"{name} RESULTS")
    print(f"{'-' * 80}")
    print(f"   Time: {comp_time:.1f}s")
    print(
        f"   Energy: E0={data['E0']:.3f} -> E={data['E'][-1]:.3f} | Error: {data['max_E_err']:.2e} ({data['max_E_err'] * 100:.3f}%)")
    print(f"   Lz:    L0={data['Lz0']:.3f} -> L={data['Lz'][-1]:.3f} | Abs Error: {data['max_Lz_err']:.2e}")
    print(f"   rho_max: {data['rho_max'][0]:.3f} -> {data['rho_max'][-1]:.3f}")

    if data['max_E_err'] < 0.01:
        quality = "EXCELLENT"
    elif data['max_E_err'] < 0.05:
        quality = "GOOD"
    elif data['max_E_err'] < 0.1:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"
    print(f"   Quality: {quality}")
    print(f"{'-' * 80}")


def main():
    print_banner()

    # Simulation 1: Single Ring
    print("\nSIMULATION 1: Single Necklace Ring")
    cfg1 = SimConfig2D(Nx=512, Ny=512, Lx=25, Ly=25, dt=0.00005, T=6.0,
                       N_peaks=6, R_ring=8.0, A_peak=0.8, w_peak=1.5,
                       g=-0.1, Omega=0.05, seed=SEED)
    sys1 = NecklaceNLSE2D(cfg1)
    psi0_1 = sys1.necklace_initial(velocity=0.0)

    t0 = time.time()
    data1 = sys1.run_simulation(psi0_1, save_full=True)
    print_results(data1, time.time() - t0, "SINGLE RING")

    create_fig1_domain(data1, sys1, cfg1)
    create_fig3_physics(data1, cfg1)
    print("Fig1 & Fig3 generated")

    # Simulation 2: Two-Ring Collision - 稍微增强非线性以提高清晰度
    print("\nSIMULATION 2: Two-Ring Collision")
    cfg2 = SimConfig2D(Nx=512, Ny=512, Lx=30, Ly=30, dt=0.00005, T=8.0,
                       N_peaks=6, R_ring=8.0, A_peak=1.0, w_peak=1.2,  # 更高振幅，更紧聚焦
                       g=-0.15, Omega=0.0, seed=SEED)  # 稍强非线性保持形状
    sys2 = NecklaceNLSE2D(cfg2)
    psi0_2 = sys2.two_ring_collision(R1=7.0, R2=14.0, v=2.5)

    t0 = time.time()
    data2 = sys2.run_simulation(psi0_2, save_full=True)
    print_results(data2, time.time() - t0, "TWO-RING COLLISION")

    create_fig2_collision(data2, sys2, cfg2)
    create_fig4_radial(data2, sys2, cfg2)
    print("Fig2 & Fig4 generated")

    # Summary
    print("\n" + "=" * 80)
    print("ALL COMPLETE!")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    max_err = max(data1['max_E_err'], data2['max_E_err'])
    print(f"Best Energy Error: {max_err:.2e}")
    print("=" * 80)


def run_parameter_scan():
    """完整定量分析 - 包含碰撞分析和g参数扫描"""
    print("\n" + "=" * 80)
    print("QUANTITATIVE ANALYSIS SUITE")
    print("=" * 80)

    # 1. 先运行基础双环模拟获取详细数据
    print("\n[1/3] Running detailed two-ring simulation...")
    cfg = SimConfig2D(Nx=512, Ny=512, Lx=30, Ly=30, dt=0.00005, T=8.0,
                      N_peaks=6, R_ring=8.0, A_peak=1.0, w_peak=1.2,
                      g=-0.15, Omega=0.0, seed=SEED)
    sys = NecklaceNLSE2D(cfg)
    psi0 = sys.two_ring_collision(R1=7.0, R2=14.0, v=2.5)

    data = sys.run_simulation(psi0, save_full=True)

    # 2. 碰撞详细分析
    print("\n[2/3] Analyzing collision dynamics...")
    collision_data = analyze_collision_details(data, sys, cfg)

    if collision_data:
        print(f"   Collision time: {collision_data['collision_time']:.3f}")
        print(f"   Min separation: {collision_data['min_separation']:.3f}")
        if collision_data['v_approach']:
            print(f"   Approach velocity: {collision_data['v_approach']:.3f}")

        # 保存JSON
        import json
        collision_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                          for k, v in collision_data.items()}
        with open(f"{OUTPUT_DIR}/collision_analysis.json", 'w') as f:
            json.dump(collision_save, f, indent=2)
        print("   Saved: collision_analysis.json")

    # 3. g参数扫描（12个点）
    print("\n[3/3] Parameter scan (12 points)...")
    print("-" * 60)
    g_values = np.linspace(-0.3, -0.05, 12)
    results = []

    for idx, g in enumerate(g_values):
        print(f"\n  Point {idx + 1}/12: g = {g:.4f}")
        cfg_s = SimConfig2D(Nx=256, Ny=256, Lx=20, Ly=20, dt=0.0001, T=4.0,
                            N_peaks=6, R_ring=6.0, A_peak=0.8, w_peak=1.2,
                            g=g, Omega=0.0, seed=SEED)
        sys_s = NecklaceNLSE2D(cfg_s)
        psi0_s = sys_s.necklace_initial()

        t_start = time.time()
        data_s = sys_s.run_simulation(psi0_s, save_full=False)
        elapsed = time.time() - t_start

        result = {
            'g': float(g),
            'E_error': float(data_s['max_E_err']),
            'Lz_error': float(data_s['max_Lz_err']),
            'E0': float(data_s['E0']),
            'E_final': float(data_s['E'][-1]),
            'rho_max_initial': float(data_s['rho_max'][0]),
            'rho_max_final': float(data_s['rho_max'][-1])
        }
        results.append(result)

        # 简洁摘要
        dE_pct = (result['E_final'] - result['E0']) / abs(result['E0']) * 100
        rho_ratio = result['rho_max_final'] / result['rho_max_initial']
        print(
            f"    Done in {elapsed:.1f}s | dE={dE_pct:+.1f}% | rho_ratio={rho_ratio:.2f} | E_err={result['E_error']:.2e}")

    print("-" * 60)

    # 保存JSON
    import json
    with open(f"{OUTPUT_DIR}/g_scan.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: g_scan.json")

    # 扫描摘要统计
    E_errs = [r['E_error'] for r in results]
    print(f"\n  Scan Summary:")
    print(f"    E_error range: {min(E_errs):.2e} ~ {max(E_errs):.2e}")
    print(f"    Mean E_error: {np.mean(E_errs):.2e}")

    # 绘制4-panel分析图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    g_vals = [r['g'] for r in results]

    # (a) 能量误差
    ax = axes[0, 0]
    E_err = [r['E_error'] * 100 for r in results]
    ax.semilogy(g_vals, E_err, 'bo-', lw=2, markersize=6)
    ax.axhline(5, color='r', ls='--', label='5% threshold')
    ax.set_xlabel('g')
    ax.set_ylabel('Energy Error (%)')
    ax.set_title('(a) Conservation vs g')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) 密度变化
    ax = axes[0, 1]
    rho_ratio = [r['rho_max_final'] / r['rho_max_initial'] for r in results]
    ax.plot(g_vals, rho_ratio, 'gs-', lw=2, markersize=6)
    ax.axhline(1, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('g')
    ax.set_ylabel(r'$\rho_{final}/\rho_{initial}$')
    ax.set_title('(b) Density Preservation')
    ax.grid(True, alpha=0.3)

    # (c) 能量漂移
    ax = axes[1, 0]
    dE = [(r['E_final'] - r['E0']) / abs(r['E0']) * 100 for r in results]
    ax.plot(g_vals, dE, 'r^-', lw=2, markersize=6)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_xlabel('g')
    ax.set_ylabel('Energy Change (%)')
    ax.set_title('(c) Energy Drift')
    ax.grid(True, alpha=0.3)

    # (d) 稳定性区域
    ax = axes[1, 1]
    stable = [1 if r['E_error'] < 0.05 else 0 for r in results]
    colors = ['green' if s else 'red' for s in stable]
    ax.scatter(g_vals, [0] * len(g_vals), c=colors, s=100, marker='s')
    ax.set_xlabel('g')
    ax.set_ylabel('Stability')
    ax.set_yticks([0])
    ax.set_yticklabels(['Status'])
    ax.set_title('(d) Stability Region (green=stable)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig5_QuantitativeAnalysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: Fig5_QuantitativeAnalysis.png")

    print("\n" + "=" * 80)
    print("QUANTITATIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print("\nGenerated:")
    print("  - collision_analysis.json")
    print("  - g_scan.json")
    print("  - Fig5_QuantitativeAnalysis.png")
    return results, collision_data


def analyze_collision_details(data, sys, cfg):
    """详细碰撞分析 - 提取时间、速度、轨迹"""
    field = data['field']
    t = data['t']

    r1_list, r2_list, sep_list = [], [], []

    for psi in field:
        r_c, rho_r = sys.radial_profile(np.abs(psi) ** 2)
        peaks, _ = find_peaks(rho_r, height=np.max(rho_r) * 0.15, distance=5)

        if len(peaks) >= 2:
            r1_list.append(r_c[peaks[0]])
            r2_list.append(r_c[peaks[1]])
            sep_list.append(abs(r_c[peaks[0]] - r_c[peaks[1]]))
        else:
            r1_list.append(np.nan)
            r2_list.append(np.nan)
            sep_list.append(np.nan)

    r1_list = np.array(r1_list)
    r2_list = np.array(r2_list)
    sep_list = np.array(sep_list)

    valid_idx = ~np.isnan(sep_list)
    if np.any(valid_idx):
        collision_idx = np.where(valid_idx)[0][np.argmin(sep_list[valid_idx])]
        collision_time = t[collision_idx]
        min_separation = sep_list[collision_idx]

        v_approach = None
        if collision_idx > 5:
            v_approach = abs(r2_list[collision_idx - 5] - r2_list[collision_idx]) / \
                         (t[collision_idx] - t[collision_idx - 5])

        return {
            'collision_time': float(collision_time),
            'min_separation': float(min_separation),
            'v_approach': float(v_approach) if v_approach else None,
            'r1_trajectory': r1_list,
            'r2_trajectory': r2_list,
            'separation': sep_list,
            't': t
        }
    return None


def main_with_analysis():
    """运行完整模拟 + 定量分析"""
    # 先运行基础模拟
    main()

    # 然后运行定量分析
    print("\n" + "=" * 80)
    print("STARTING QUANTITATIVE ANALYSIS...")
    print("=" * 80)
    run_parameter_scan()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        # 仅运行参数扫描
        run_parameter_scan()
    elif len(sys.argv) > 1 and sys.argv[1] == "--basic":
        # 仅运行基础模拟
        main()
    else:
        # 默认：运行完整模拟 + 定量分析
        main_with_analysis()
