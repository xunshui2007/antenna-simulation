"""
2D FDTD Simulation of a Center-Fed Dipole Antenna
TMz formulation: Ez, Hx, Hy components (Yee grid)
- Perfect Electric Conductor (PEC) dipole arms
- First-order Mur absorbing boundary conditions
- Gaussian pulse excitation at dipole feed gap
- Real-time visualization of Ez field
- Far-field radiation pattern calculation
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, patches
from matplotlib.axes import Axes
from typing import Tuple, Optional


class FDTDDipole:
    """2D FDTD simulation of a center-fed dipole antenna (TMz)."""

    def __init__(
        self,
        nx: int = 160,
        ny: int = 160,
        dx: float = 1.0,
        dy: float = 1.0,
        dipole_length: int = 30,
        feed_gap: int = 4,
        courant: float = 0.5,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.courant = courant

        c = 1.0
        dt_max = 1.0 / (c * np.sqrt(1 / dx ** 2 + 1 / dy ** 2))
        self.dt = courant * dt_max

        self.dipole_length = dipole_length
        self.feed_gap = feed_gap
        self.feed_x = nx // 2
        self.feed_y = ny // 2
        self.time_step = 0

        self._init_fields()
        self._init_pec()
        self._init_source_positions()
        self._init_abc()

    def _init_fields(self):
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny))
        self.Hy = np.zeros((self.nx, self.ny))

    def _init_pec(self):
        half_len = self.dipole_length // 2
        gap_half = self.feed_gap // 2
        dipole_start = self.feed_y - half_len
        dipole_end = self.feed_y + half_len
        gap_start = self.feed_y - gap_half
        gap_end = self.feed_y + gap_half

        self.pec_mask = np.zeros((self.nx, self.ny), dtype=bool)
        for i in range(self.feed_x, self.feed_x + 3):
            if i >= self.nx:
                break
            for j in range(dipole_start, dipole_end + 1):
                if 0 <= j < self.ny and not (gap_start <= j <= gap_end):
                    self.pec_mask[i, j] = True

    def _init_source_positions(self):
        gap_half = self.feed_gap // 2
        self.source_positions = []
        for j in range(self.feed_y - gap_half, self.feed_y + gap_half + 1):
            if 0 <= j < self.ny:
                self.source_positions.append((self.feed_x, j))

    def _init_abc(self):
        coeff = (self.courant - 1) / (self.courant + 1)
        self.abc_coeff = coeff
        self.Ez_prev = np.zeros((self.nx, self.ny))

    def gaussian_pulse(self, t: int, delay: float = 40.0, width: float = 12.0) -> float:
        return np.exp(-((t - delay) / width) ** 2)

    def step(self):
        self.Hx[:, :-1] -= self.dt / self.dy * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:-1, :] += self.dt / self.dx * (self.Ez[1:, :] - self.Ez[:-1, :])

        curl_h = np.zeros_like(self.Ez)
        curl_h[1:-1, 1:-1] = (
            (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy
        )
        self.Ez += self.dt * curl_h

        source_val = self.gaussian_pulse(self.time_step)
        for i, j in self.source_positions:
            self.Ez[i, j] = source_val

        self.Ez[self.pec_mask] = 0.0
        self._apply_mur_abc()
        self.time_step += 1

    def _apply_mur_abc(self):
        coeff = self.abc_coeff
        nx, ny = self.nx, self.ny
        for j in range(1, ny - 1):
            self.Ez[0, j] = self.Ez_prev[1, j] + coeff * (self.Ez[1, j] - self.Ez[0, j])
            self.Ez[-1, j] = self.Ez_prev[-2, j] + coeff * (self.Ez[-2, j] - self.Ez[-1, j])
        for i in range(1, nx - 1):
            self.Ez[i, 0] = self.Ez_prev[i, 1] + coeff * (self.Ez[i, 1] - self.Ez[i, 0])
            self.Ez[i, -1] = self.Ez_prev[i, -2] + coeff * (self.Ez[i, -2] - self.Ez[i, -1])
        self.Ez_prev[:] = self.Ez

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def get_radiation_pattern(
        self, radius: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if radius is None:
            radius = min(self.nx, self.ny) // 3
        cx, cy = self.feed_x, self.feed_y
        angles = np.linspace(0, 2 * np.pi, 360)
        pattern = np.zeros_like(angles)

        for k, theta in enumerate(angles):
            ix = int(round(cx + radius * np.cos(theta)))
            iy = int(round(cy + radius * np.sin(theta)))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                pattern[k] = abs(self.Ez[ix, iy])

        max_val = pattern.max()
        if max_val > 1e-15:
            pattern /= max_val
        return angles, pattern

    def simulate_with_animation(
        self,
        n_steps: int = 400,
        save_path: Optional[str] = None,
        interval: int = 30,
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_xlabel("x (cells)")
        ax1.set_ylabel("y (cells)")
        ax1.set_title("Ez Field (Dipole Antenna)")
        ax1.set_aspect("equal")

        im = ax1.imshow(
            self.Ez.T,
            cmap="RdBu_r",
            origin="lower",
            extent=[0, self.nx * self.dx, 0, self.ny * self.dy],
            vmin=-0.5,
            vmax=0.5,
            animated=True,
        )
        plt.colorbar(im, ax=ax1, label="Ez")
        self._add_dipole_patch(ax1)

        pattern_angles = np.linspace(0, 2 * np.pi, 360)
        (pattern_line,) = ax2.plot(pattern_angles, np.zeros(360), "b-", linewidth=1.5)
        ax2.set_ylim(0, 1.2)
        ax2.set_xlim(0, 2 * np.pi)
        ax2.set_xlabel("Angle (rad)")
        ax2.set_ylabel("Normalized |Ez|")
        ax2.set_title("Radiation Pattern")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

        time_text = ax1.text(
            0.02, 0.95, "", transform=ax1.transAxes, color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5),
        )

        def animate(frame):
            for _ in range(2):
                self.step()
            vmax = max(abs(self.Ez).max(), 1e-10)
            im.set_clim(-vmax, vmax)
            im.set_array(self.Ez.T)
            time_text.set_text(f"Step: {self.time_step}")
            if self.time_step > 50:
                angles, pat = self.get_radiation_pattern()
                pattern_line.set_data(angles, pat)
            return im, pattern_line, time_text

        anim = animation.FuncAnimation(fig, animate, frames=n_steps // 2, interval=interval, blit=True)
        if save_path:
            anim.save(save_path, writer="pillow", fps=30)
        plt.tight_layout()
        plt.show()
        return anim

    def _add_dipole_patch(self, ax: Axes):
        half_len = self.dipole_length // 2
        gap_half = self.feed_gap // 2
        ds = self.feed_y - half_len
        de = self.feed_y + half_len
        gs = self.feed_y - gap_half
        ge = self.feed_y + gap_half
        x0 = self.feed_x * self.dx - 0.5
        for y0, h, c in [(ge, de - ge, "yellow"), (ds, gs - ds, "yellow"), (gs, ge - gs, "red")]:
            ax.add_patch(patches.Rectangle(
                (x0, y0 * self.dy), 2, h * self.dy,
                linewidth=1, edgecolor=c, facecolor=c, alpha=0.3 if c == "yellow" else 0.5,
            ))

    def plot_final_state(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        titles = ["Ez Field", "Hx Field", "Hy Field"]
        data = [self.Ez.T, self.Hx.T, self.Hy.T]
        for ax, t, d in zip(axes, titles, data):
            im = ax.imshow(d, cmap="RdBu_r", origin="lower",
                           extent=[0, self.nx * self.dx, 0, self.ny * self.dy])
            ax.set_title(t)
            ax.set_xlabel("x (cells)")
            ax.set_ylabel("y (cells)")
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_radiation_pattern(self, save_path: Optional[str] = None):
        angles, pattern = self.get_radiation_pattern()
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"},
        )
        ax1.plot(angles, pattern, "b-", linewidth=1.5)
        ax1.set_title("Radiation Pattern (Polar)")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax2.plot(angles, 10 * np.log10(pattern + 1e-15), "r-", linewidth=1.5)
        ax2.set_title("Radiation Pattern (dB)")
        ax2.set_ylim(-40, 3)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def main():
    sim = FDTDDipole(nx=160, ny=160, dipole_length=30, feed_gap=4)
    sim.simulate_with_animation(n_steps=400)
    sim.run(200)
    sim.plot_final_state(save_path="fdtd_fields.png")
    sim.plot_radiation_pattern(save_path="radiation_pattern.png")
    print(f"Completed: {sim.time_step} steps")


if __name__ == "__main__":
    main()
