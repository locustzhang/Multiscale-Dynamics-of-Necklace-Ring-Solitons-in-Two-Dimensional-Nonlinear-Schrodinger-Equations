#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ⚠ CRITICAL: Allocator config BEFORE torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"

import json
import time
import random
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from tqdm import tqdm

try:
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.backends.cuda.cufft_plan_cache.max_size = 1
except Exception:
    pass

warnings.filterwarnings("ignore")

# =============================================================================
# 【所有参数只改这里】
# =============================================================================
SEED            = 42
OUTPUT_DIR      = "Necklace_Soliton_HIGH_PRECISION_FULL"

NX_SINGLE       = 1024
NY_SINGLE       = 1024
LX_SINGLE       = 25.0
LY_SINGLE       = 25.0
T_SINGLE        = 6.0
G_SINGLE        = -0.08
V_TRAP_SINGLE   = 0.001

NX_COLL         = 1024
NY_COLL         = 1024
LX_COLL         = 30.0
LY_COLL         = 30.0
T_COLL          = 8.0
G_COLL          = -0.08
V_TRAP_COLL     = 0.001

N_PEAKS         = 6
R_RING          = 8.0
A_PEAK          = 0.8
W_PEAK          = 1.5
OMEGA_ROT       = 0.0
DT_DEFAULT      = 1e-5

SCAN_G_MIN      = -0.30
SCAN_G_MAX      = -0.04
SCAN_N_POINTS   = 14

# =============================================================================
# Journal Style
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.grid': False,
    'axes.linewidth': 1.0,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'savefig.dpi': 600,
})

SC_COLORS = {
    'primary': '#003049',
    'energy': '#d62828',
    'norm': '#f77f00',
    'lz': '#fcbf49',
    'density': '#eae2b7',
    'accent': '#457b9d'
}

# =============================================================================
# 绘图工具
# =============================================================================
_cmap_data = [
    (0.05, 0.05, 0.20),
    (0.10, 0.30, 0.65),
    (0.95, 0.95, 0.95),
    (0.85, 0.25, 0.10),
    (0.55, 0.05, 0.05),
]
DENSITY_CMAP = LinearSegmentedColormap.from_list('soliton_density', _cmap_data, N=512)

def style_ax(ax, minor=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(direction='in', width=0.7, length=4, which='major')
    ax.tick_params(direction='in', width=0.5, length=2, which='minor')
    if minor:
        ax.minorticks_on()

def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='left', fontfamily='serif')

def add_shared_colorbar(fig, im, label=r'Density $|\psi|^2$',
                        rect=(0.93, 0.3, 0.015, 0.4), fontsize=12):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize)
    cb.outline.set_visible(False)
    return cb

def _density_panel(ax, z, t_val, extent, letter,
                   highlight=False, gamma=0.45, vmin_floor=1e-4):
    """统一密度面板：magma + PowerNorm，与 Fig2 风格一致。"""
    rho = np.abs(z).T ** 2
    im = ax.imshow(
        rho,
        origin='lower',
        cmap=plt.cm.magma,
        norm=PowerNorm(gamma=gamma, vmin=vmin_floor,
                       vmax=max(rho.max(), vmin_floor * 10)),
        extent=extent,
        interpolation='bicubic',
    )
    ax.text(0.05, 0.90, fr'$t={t_val:.1f}$',
            transform=ax.transAxes, color='white',
            fontsize=10, fontweight='bold', va='top')
    ax.set_xticks([])
    ax.set_yticks([])
    panel_label(ax, letter, x=0.02, y=1.02)
    if highlight:
        for spine in ax.spines.values():
            spine.set_edgecolor('#ff4d6d')
            spine.set_linewidth(2.2)
    return im

# =============================================================================
# Seed & Device
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Config
# =============================================================================
@dataclass
class SimConfig:
    Nx: int = 512
    Ny: int = 512
    Lx: float = 25.0
    Ly: float = 25.0
    dt: float = DT_DEFAULT
    T: float = 6.0
    N_peaks: int = N_PEAKS
    R_ring: float = R_RING
    A_peak: float = A_PEAK
    w_peak: float = W_PEAK
    g: float = -0.08
    V_trap: float = 0.0
    Omega: float = OMEGA_ROT
    seed: int = SEED
    dtype_real: torch.dtype = torch.float64
    dtype_cplx: torch.dtype = torch.complex128

    def __post_init__(self):
        self.dx = 2 * self.Lx / self.Nx
        self.dy = 2 * self.Ly / self.Ny
        self.x = (torch.arange(self.Nx, device=DEVICE, dtype=self.dtype_real) - self.Nx // 2) * self.dx
        self.y = (torch.arange(self.Ny, device=DEVICE, dtype=self.dtype_real) - self.Ny // 2) * self.dy

    @property
    def N_cr(self):
        # Townes soliton threshold（仅对无势均匀空间成立）
        return 5.85 / abs(self.g)

    def validate(self, N_total: float) -> dict:
        cfl_limit = self.dx ** 2 / (2 * np.pi**2)   # 修正：Nyquist 精确上界
        nl_phase = abs(self.g) * (self.A_peak ** 2) * self.dt
        return {
            'CFL_satisfied': self.dt < cfl_limit,
            'dt_over_CFL': self.dt / cfl_limit,
            'NL_phase_per_step': nl_phase,
            'N_total': N_total,
            'N_cr': self.N_cr,
            'collapse_safe': N_total < self.N_cr,
            'N_over_Ncr': N_total / self.N_cr,
        }

    def print_config(self):
        print(f"  Config: Nx={self.Nx}, Ny={self.Ny}, L={self.Lx}, dx={self.dx:.4f}")
        print(f"  Soliton: N_peaks={self.N_peaks}, R_ring={self.R_ring}, g={self.g}, V_trap={self.V_trap}")
        print(f"  Precision: float64 + complex128 [HIGH PRECISION]")
        print(f"  Stability: N_cr={self.N_cr:.4f} (uniform, no trap)")

# =============================================================================
# Snapshot Manager
# =============================================================================
class SnapshotManager:
    def __init__(self, base_dir: str, label: str):
        self.label = label
        self.snap_dir = os.path.join(base_dir, f"Snapshots_{label}")
        os.makedirs(self.snap_dir, exist_ok=True)
        self.saved_files: List[str] = []

    def save(self, psi: torch.Tensor):
        fname = f"snap_{len(self.saved_files):05d}.npy"
        path = os.path.join(self.snap_dir, fname)
        # ✅ 修正：保留 complex128，不降精度
        arr = psi.detach().cpu().numpy().astype(np.complex128)
        np.save(path, arr)
        self.saved_files.append(path)

    def load(self, idx: int) -> np.ndarray:
        safe_idx = min(idx, len(self.saved_files) - 1)
        return np.load(self.saved_files[safe_idx], mmap_mode='r')

    def save_meta(self, meta: dict):
        with open(os.path.join(self.snap_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)

# =============================================================================
# GPE Solver
# =============================================================================
class GPESolver:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.X, self.Y = torch.meshgrid(cfg.x, cfg.y, indexing='ij')
        kx = torch.fft.fftfreq(cfg.Nx, d=cfg.dx).to(DEVICE, dtype=torch.float64) * (2 * np.pi)
        ky = torch.fft.fftfreq(cfg.Ny, d=cfg.dy).to(DEVICE, dtype=torch.float64) * (2 * np.pi)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.V = 0.5 * self.cfg.V_trap * (self.X**2 + self.Y**2)
        self.prop_half = torch.exp((-1j) * self.K2 * (cfg.dt / 4)).to(torch.complex128)
        self.R_cpu = torch.sqrt(self.X**2 + self.Y**2).detach().cpu().numpy()
        self.R_flat = self.R_cpu.ravel()

    def necklace_initial(self) -> torch.Tensor:
        cfg = self.cfg
        psi = torch.zeros((cfg.Nx, cfg.Ny), dtype=torch.complex128, device=DEVICE)
        angles = torch.linspace(0, 2 * np.pi, cfg.N_peaks + 1, device=DEVICE)[:-1]
        for th in angles:
            x0 = cfg.R_ring * torch.cos(th)
            y0 = cfg.R_ring * torch.sin(th)
            r2 = (self.X - x0)**2 + (self.Y - y0)**2
            psi += cfg.A_peak * torch.exp(-r2 / cfg.w_peak**2)
        if cfg.Omega != 0:
            phi = torch.atan2(self.Y, self.X)
            psi *= torch.exp(1j * cfg.Omega * phi)
        return psi

    def two_ring_collision(self, R1=7.0, R2=11.0, v=1.5):
        """
        两环碰撞初态。
        包络：以各峰中心为参考的高斯函数。
        相位：e^{±ivR}，R 为到原点的径向距离，编码内环向外(+v)、外环向内(-v)的群速度。
        """
        cfg = self.cfg
        psi = torch.zeros_like(self.X, dtype=torch.complex128, device=DEVICE)
        angles = torch.linspace(0, 2 * np.pi, cfg.N_peaks + 1, device=DEVICE)[:-1]
        # ✅ 修正：相位用到原点的径向距离，与环的运动方向物理自洽
        R_origin = torch.sqrt(self.X**2 + self.Y**2 + 1e-12)
        for th in angles:
            # 内环：向外传播 +v
            x0, y0 = R1 * torch.cos(th), R1 * torch.sin(th)
            r2 = (self.X - x0)**2 + (self.Y - y0)**2
            psi += cfg.A_peak * torch.exp(-r2 / cfg.w_peak**2) * torch.exp( 1j * v * R_origin)
            # 外环：向内传播 -v
            x0, y0 = R2 * torch.cos(th), R2 * torch.sin(th)
            r2 = (self.X - x0)**2 + (self.Y - y0)**2
            psi += cfg.A_peak * torch.exp(-r2 / cfg.w_peak**2) * torch.exp(-1j * v * R_origin)
        return psi

    @torch.no_grad()
    def diagnostics(self, psi):
        cfg = self.cfg
        rho = torch.abs(psi)**2
        N = torch.sum(rho).item() * cfg.dx * cfg.dy
        rho_max = torch.max(rho).item()

        psi_k = torch.fft.fft2(psi, norm="ortho")
        dpx = torch.fft.ifft2(psi_k * (1j * self.KX), norm="ortho")
        dpy = torch.fft.ifft2(psi_k * (1j * self.KY), norm="ortho")
        del psi_k

        Ekin  = 0.5 * torch.sum(torch.abs(dpx)**2 + torch.abs(dpy)**2).item() * cfg.dx * cfg.dy
        Eint  = 0.5 * cfg.g * torch.sum(rho**2).item() * cfg.dx * cfg.dy
        Etrap = torch.sum(self.V * rho).item() * cfg.dx * cfg.dy
        Etot  = Ekin + Eint + Etrap
        Lz    = torch.sum(torch.conj(psi) * (-1j) * (self.X * dpy - self.Y * dpx)).real.item() * cfg.dx * cfg.dy
        del dpx, dpy
        return Etot, Ekin, Eint, Etrap, N, Lz, rho_max

    @torch.no_grad()
    def step(self, psi):
        psi_k = torch.fft.fft2(psi, norm="ortho")
        psi_k.mul_(self.prop_half)
        psi = torch.fft.ifft2(psi_k, norm="ortho")
        del psi_k

        rho = torch.abs(psi)**2
        psi.mul_(torch.exp((-1j) * self.cfg.dt * (self.cfg.g * rho + self.V)))

        psi_k = torch.fft.fft2(psi, norm="ortho")
        psi_k.mul_(self.prop_half)
        psi = torch.fft.ifft2(psi_k, norm="ortho")
        del psi_k
        return psi

    def radial_profile(self, rho_np, n_bins=120):
        R = self.R_flat
        rho_flat = rho_np.ravel()
        r_bins = np.linspace(0, self.cfg.Lx, n_bins + 1)
        rho_r, _ = np.histogram(R, bins=r_bins, weights=rho_flat)
        counts, _ = np.histogram(R, bins=r_bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_r = np.divide(rho_r, counts, where=counts > 0)
        r_cen = 0.5 * (r_bins[:-1] + r_bins[1:])
        return r_cen, rho_r

    def run(self, psi0, label="sim", save_fields=True, n_save=200):
        cfg = self.cfg
        steps = int(cfg.T / cfg.dt)
        interval = max(1, steps // n_save)
        psi = psi0.clone()
        snap = SnapshotManager(OUTPUT_DIR, label) if save_fields else None

        E0, Ek0, Ei0, Etrap0, N0, Lz0, rho0 = self.diagnostics(psi)
        stab = cfg.validate(N0)
        print(f"\n[{label}] N0={N0:.4f}, N/Ncr={stab['N_over_Ncr']:.4f} | "
              f"{'SAFE' if stab['collapse_safe'] else 'UNSAFE'}")

        data = {
            't': [], 'E': [], 'E_kin': [], 'E_int': [], 'E_trap': [],
            'N': [], 'Lz': [], 'rho_max': [],
            'E_err': [], 'N_err': [], 'Lz_err': [],
            'max_E_err': 0, 'max_N_err': 0, 'max_Lz_err': 0,
            'E0': E0, 'N0': N0, 'Lz0': Lz0, 'stab': stab,
            'collapsed': False, 'label': label
        }

        pbar = tqdm(range(steps), desc=label, colour='blue')
        for i in pbar:
            psi = self.step(psi)
            if i % interval == 0:
                t_now = i * cfg.dt
                E, Ek, Ei, Etrap, N, Lz, rho_max = self.diagnostics(psi)
                data['t'].append(t_now)
                data['E'].append(E)
                data['E_kin'].append(Ek)
                data['E_int'].append(Ei)
                data['E_trap'].append(Etrap)
                data['N'].append(N)
                data['Lz'].append(Lz)
                data['rho_max'].append(rho_max)
                data['E_err'].append(abs(E - E0) / (abs(E0) + 1e-30))
                data['N_err'].append(abs(N - N0) / (abs(N0) + 1e-30))
                data['Lz_err'].append(abs(Lz - Lz0))  # 绝对误差，避免 Lz0≈0 时发散
                if save_fields:
                    snap.save(psi)
                if rho_max > 100 * rho0:
                    data['collapsed'] = True
                    break
                pbar.set_postfix(E=f"{data['E_err'][-1]:.1e}",
                                 N=f"{data['N_err'][-1]:.1e}")
            if i % 4000 == 0 and DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        if save_fields:
            snap.save(psi)
            data['snap_files'] = snap.saved_files
        else:
            data['snap_files'] = []
        pbar.close()

        for k in ['t', 'E', 'E_kin', 'E_int', 'E_trap', 'N', 'Lz',
                  'rho_max', 'E_err', 'N_err', 'Lz_err']:
            data[k] = np.array(data[k])

        data['max_E_err']  = data['E_err'].max()
        data['max_N_err']  = data['N_err'].max()
        data['max_Lz_err'] = data['Lz_err'].max()
        data['rho_ratio']  = data['rho_max'][-1] / data['rho_max'][0]

        print(f"\n[{label}] Run Summary (HIGH PRECISION):")
        print(f"  - Max Energy Error: {data['max_E_err']:.2e}")
        print(f"  - Max Norm Error:   {data['max_N_err']:.2e}")
        print(f"  - Max |ΔLz|:        {data['max_Lz_err']:.2e}")
        print(f"  - Peak Density:     {rho0:.4f} -> {data['rho_max'][-1]:.4f} "
              f"(ratio: {data['rho_ratio']:.3f})")
        return data

# =============================================================================
# 四张论文图
# =============================================================================
C = dict(red='#c0392b', orange='#d46a00', teal='#0e7f8a', blue='#1f4e8c')

# ---------------------------------------------------------------------------
# Fig 1：单环孤子时空演化（2×2，四个关键时刻）
# ---------------------------------------------------------------------------
def fig1_single_ring(data, cfg):
    t = data['t']
    snap_files = data['snap_files']
    n = len(snap_files)
    idx = [0, n // 3, 2 * n // 3, n - 1]
    extent = [cfg.x.min().item(), cfg.x.max().item(),
              cfg.y.min().item(), cfg.y.max().item()]

    fig = plt.figure(figsize=(9, 8.5))
    gs = GridSpec(2, 2, wspace=0.08, hspace=0.18,
                  left=0.05, right=0.90, top=0.92, bottom=0.05)
    im_last = None
    for i, (k, letter) in enumerate(zip(idx, 'abcd')):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        z = np.load(snap_files[k])
        tidx = min(k, len(t) - 1)
        im_last = _density_panel(ax, z, t[tidx], extent, letter)

    add_shared_colorbar(fig, im_last, rect=(0.92, 0.20, 0.02, 0.60))
    fig.suptitle('Fig. 1. Spatiotemporal evolution of necklace soliton',
                 fontsize=14, y=0.97)
    fig.savefig(f'{OUTPUT_DIR}/Fig1_SingleRing.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Fig1] saved.")

# ---------------------------------------------------------------------------
# Fig 2：双环碰撞序列（2×3，精选6帧，去掉冗余中间帧）
# ---------------------------------------------------------------------------
def fig2_collision(data, cfg):
    t = data['t']
    snap_files = data['snap_files']
    n = len(snap_files)
    # 精选6帧：碰前、接近、碰撞、穿透、分离、末态
    idx = np.linspace(0, n - 1, 6, dtype=int)
    extent = [cfg.x.min().item(), cfg.x.max().item(),
              cfg.y.min().item(), cfg.y.max().item()]
    clip = 0.70

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, wspace=0.08, hspace=0.18,
                  left=0.04, right=0.91, top=0.93, bottom=0.04)
    im_last = None
    # 高亮第3帧（碰撞时刻，idx=2）
    highlight_i = 2
    for i, (k, letter) in enumerate(zip(idx, 'abcdef')):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        z = np.load(snap_files[k])
        tidx = min(k, len(t) - 1)
        im_last = _density_panel(ax, z, t[tidx], extent, letter,
                                  highlight=(i == highlight_i))
        ax.set_xlim(-cfg.Lx * clip, cfg.Lx * clip)
        ax.set_ylim(-cfg.Ly * clip, cfg.Ly * clip)

    add_shared_colorbar(fig, im_last)
    fig.suptitle('Fig. 2. Nonlinear collision dynamics of double necklace solitons',
                 fontsize=14, y=0.96)
    fig.savefig(f'{OUTPUT_DIR}/Fig2_Collision.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Fig2] saved.")

# ---------------------------------------------------------------------------
# Fig 3：守恒量误差（单环 + 双环合并，4子图，不同线型区分）
# ---------------------------------------------------------------------------
def fig3_conservation(dat1, dat2):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplots_adjust(wspace=0.28, hspace=0.35)

    pairs = [
        ('E_err',   r'Rel. energy error $|\Delta E/E_0|$',   True),
        ('N_err',   r'Rel. norm error $|\Delta N/N_0|$',     True),
        ('Lz_err',  r'Ang. mom. error $|\Delta L_z|$', True),
        ('rho_max', r'Peak density $\rho_\mathrm{max}$',     False),
    ]

    for ax, (key, ylabel, logy), letter in zip(axs.flat, pairs, 'abcd'):
        ax.plot(dat1['t'], dat1[key], color=C['blue'],
                lw=1.8, ls='-',  label='Single ring')
        ax.plot(dat2['t'], dat2[key], color=C['red'],
                lw=1.8, ls='--', label='Double ring')
        if logy:
            ax.set_yscale('log')
        if key == 'rho_max':
            ax.fill_between(dat1['t'], dat1[key], alpha=0.10, color=C['blue'])
            ax.fill_between(dat2['t'], dat2[key], alpha=0.10, color=C['red'])
        ax.set_xlabel('Time $t$')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)
        style_ax(ax)
        panel_label(ax, letter)

    fig.suptitle('Fig. 3. Numerical conservation and peak density', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'{OUTPUT_DIR}/Fig3_Conservation.pdf')
    plt.close()
    print(f"[Fig3] saved.")

# ---------------------------------------------------------------------------
# Fig 4：径向时空图（单环）+ 参数扫描（g 扫描稳定性）
# ---------------------------------------------------------------------------
def fig4_stability(data, solver, cfg, scan_res):
    t = data['t']
    snap_files = data['snap_files']

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.32,
                  left=0.08, right=0.96, top=0.88, bottom=0.13)

    # --- 左：径向时空图（kymograph）---
    ax1 = fig.add_subplot(gs[0])
    r_mat, t_plot = [], []
    step = max(1, len(snap_files) // 200)
    for i in range(0, len(snap_files), step):
        z = np.load(snap_files[i])
        rc, rr = solver.radial_profile(np.abs(z)**2)
        r_mat.append(rr)
        tidx = min(i, len(t) - 1)
        t_plot.append(t[tidx])
    r_mat = np.array(r_mat)
    im = ax1.imshow(
        r_mat, aspect='auto', origin='lower',
        cmap=plt.cm.magma,
        norm=PowerNorm(gamma=0.45, vmin=max(r_mat.min(), 1e-4), vmax=r_mat.max()),
        extent=[rc[0], rc[-1], t_plot[0], t_plot[-1]],
        interpolation='bicubic',
    )
    ax1.set_xlabel('Radius $r$')
    ax1.set_ylabel('Time $t$')
    cb1 = plt.colorbar(im, ax=ax1)
    cb1.set_label(r'$\langle\rho\rangle_\theta$', fontsize=11)
    cb1.outline.set_visible(False)
    style_ax(ax1)
    panel_label(ax1, 'a')

    # --- 右：参数扫描，g vs 能量误差，标注稳定区间 ---
    ax2 = fig.add_subplot(gs[1])
    g_vals    = np.array([r['g']         for r in scan_res])
    e_errs    = np.array([r['E_error']   for r in scan_res])
    n_over_nc = np.array([r['N_over_Ncr'] for r in scan_res])

    sc = ax2.scatter(g_vals, e_errs, c=n_over_nc,
                     cmap='plasma', s=48, zorder=3, edgecolors='none')
    ax2.plot(g_vals, e_errs, color='grey', lw=0.8, alpha=0.5, zorder=2)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Interaction strength $g$')
    ax2.set_ylabel(r'Max rel. energy error $|\Delta E/E_0|$')
    ax2.axvline(G_SINGLE, color=C['teal'], lw=1.2, ls='--',
                label=fr'$g={G_SINGLE}$ (main sim)')
    ax2.legend(fontsize=9)
    cb2 = plt.colorbar(sc, ax=ax2)
    cb2.set_label(r'$N_0/N_\mathrm{cr}$', fontsize=10)
    cb2.outline.set_visible(False)
    style_ax(ax2)
    panel_label(ax2, 'b')

    fig.suptitle('Fig. 4. Radial stability and parameter scan', fontsize=14)
    fig.savefig(f'{OUTPUT_DIR}/Fig4_Stability.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Fig4] saved.")

# =============================================================================
# Scan & Table
# =============================================================================
def run_scan():
    g_list = np.linspace(SCAN_G_MIN, SCAN_G_MAX, SCAN_N_POINTS)
    res = []
    print(f"\n[Parameter Scan] {SCAN_N_POINTS} points ...")
    for g in g_list:
        cfg = SimConfig(Nx=256, Ny=256, Lx=20, Ly=20,
                        dt=1e-4, T=3, g=float(g), V_trap=V_TRAP_SINGLE)
        sol = GPESolver(cfg)
        psi = sol.necklace_initial()
        dat = sol.run(psi, f'scan_g{g:.3f}', save_fields=False, n_save=80)
        res.append({
            'g':          float(g),
            'E_error':    float(dat['max_E_err']),
            'N_error':    float(dat['max_N_err']),
            'rho_ratio':  float(dat['rho_ratio']),
            'N_over_Ncr': float(dat['N0'] / cfg.N_cr),
        })
        del sol, psi
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    with open(f'{OUTPUT_DIR}/scan.json', 'w') as f:
        json.dump(res, f, indent=2)
    return res

def print_summary(datas, names):
    print("\n" + "=" * 90)
    print("LATEX SUMMARY TABLE")
    print("=" * 90)
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\hline")
    print(r"Run & $N_0$ & $N_0/N_\mathrm{cr}$ & $\max|\Delta E/E_0|$ "
          r"& $\max|\Delta N/N_0|$ & $\rho_\mathrm{max}/\rho_0$ & Collapsed \\")
    print(r"\hline")
    for d, n in zip(datas, names):
        print(f"{n} & {d['N0']:.4f} & {d['N0']/d['stab']['N_cr']:.4f} & "
              f"{d['max_E_err']:.2e} & {d['max_N_err']:.2e} & "
              f"{d['rho_ratio']:.3f} & {'Yes' if d['collapsed'] else 'No'} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print("=" * 90)

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("NECKLACE SOLITON — FULL HIGH PRECISION (complex128)")
    print(f"DEVICE: {DEVICE} | OUTPUT: {OUTPUT_DIR}")
    print("=" * 70)
    t0 = time.time()

    # --- Single ring ---
    print("\n--- Single Necklace Soliton ---")
    cfg1 = SimConfig(Nx=NX_SINGLE, Ny=NY_SINGLE, Lx=LX_SINGLE, Ly=LY_SINGLE,
                     T=T_SINGLE, g=G_SINGLE, V_trap=V_TRAP_SINGLE)
    cfg1.print_config()
    sol1 = GPESolver(cfg1)
    psi1 = sol1.necklace_initial()
    dat1 = sol1.run(psi1, "SingleRing", n_save=200)
    fig1_single_ring(dat1, cfg1)
    del psi1
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # --- Two-ring collision ---
    print("\n--- Two-Ring Collision ---")
    cfg2 = SimConfig(Nx=NX_COLL, Ny=NY_COLL, Lx=LX_COLL, Ly=LY_COLL,
                     T=T_COLL, g=G_COLL, V_trap=V_TRAP_COLL)
    cfg2.print_config()
    sol2 = GPESolver(cfg2)
    psi2 = sol2.two_ring_collision()
    dat2 = sol2.run(psi2, "TwoRing", n_save=240)
    fig2_collision(dat2, cfg2)
    del psi2
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # --- Fig3: 合并守恒量图 ---
    fig3_conservation(dat1, dat2)

    # --- 参数扫描 + Fig4 ---
    scan = run_scan()
    fig4_stability(dat1, sol1, cfg1, scan)

    del sol1, sol2
    print_summary([dat1, dat2], ["Single Ring", "Two-Ring Collision"])
    print(f"\n✅ ALL DONE in {(time.time() - t0) / 60:.2f} min")

if __name__ == "__main__":
    main()
