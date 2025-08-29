#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def spectral_wavenumbers(N, L):
    # Radian wavenumbers
    k1d = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1e-12
    # Integer frequency indices for de-aliasing
    n1d = np.fft.fftfreq(N) * N
    nX, nY, nZ = np.meshgrid(n1d, n1d, n1d, indexing='ij')
    return (KX, KY, KZ, K2), (nX, nY, nZ)

def two_thirds_mask(nX, nY, nZ, N):
    cutoff = N//3
    return (np.abs(nX) < cutoff) & (np.abs(nY) < cutoff) & (np.abs(nZ) < cutoff)

def velocity_from_omega(omega_hat, KX, KY, KZ, K2):
    ox, oy, oz = omega_hat
    ikx, iky, ikz = 1j*KX, 1j*KY, 1j*KZ
    cx = iky*oz - ikz*oy
    cy = ikz*ox - ikx*oz
    cz = ikx*oy - iky*ox
    ux = cx / K2; uy = cy / K2; uz = cz / K2
    return np.array([ux, uy, uz])

def L4norm_of_magnitude(vec, L):
    dx3 = (L / vec.shape[1])**3
    mag = np.linalg.norm(vec, axis=0)
    return float(np.sum(np.abs(mag)**4) * dx3)

def L2norm_of_magnitude(vec, L):
    dx3 = (L / vec.shape[1])**3
    mag = np.linalg.norm(vec, axis=0)
    return float(np.sum(np.abs(mag)**2) * dx3)

def grad_u(u, L):
    dx = L / u.shape[1]
    gu = np.zeros((3,3) + u.shape[1:], dtype=u.dtype)
    for j in range(3):
        for k in range(3):
            gu[j,k] = np.gradient(u[j], axis=k) / dx
    return gu

def laplacian_spectral(field, K2):
    hat = np.fft.fftn(field, axes=(1,2,3))
    lap_hat = -K2 * hat
    lap = np.fft.ifftn(lap_hat, axes=(1,2,3)).real
    return lap

def save_csv(rows, path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

class SpectralNS:
    def __init__(self, N=64, L=2*np.pi, nu=1e-3, R=0.5, A=3.0, a0=1.0, dt=2.5e-3):
        self.N, self.L, self.nu, self.R, self.A, self.a0, self.dt = N, L, nu, R, A, a0, dt
        self.x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.x, self.x, indexing='ij')
        (self.KX, self.KY, self.KZ, self.K2), (self.nX, self.nY, self.nZ) = spectral_wavenumbers(N, L)
        self.mask = two_thirds_mask(self.nX, self.nY, self.nZ, N)
        self._init_vorticity()
        self.g = self.bump_swirl_force(A=self.A, R=self.R)

    def _init_vorticity(self):
        N, L = self.N, self.L
        X, Y, Z = self.X, self.Y, self.Z
        xc = yc = zc = L/2
        r = np.sqrt((X-xc)**2 + (Y-yc)**2 + (Z-zc)**2)
        core = (r < self.R/2)
        omega = np.zeros((3,N,N,N), dtype=np.float64)
        omega[2][core] = self.a0
        self.omega = omega

    def bump_swirl_force(self, A=3.0, R=0.5):
        X, Y, Z = self.X, self.Y, self.Z
        L = self.L
        dx = L / self.N
        xc = yc = zc = L/2
        r_plane = np.sqrt((X-xc)**2 + (Y-yc)**2)
        chi_r = np.zeros_like(r_plane); chi_r[r_plane < R] = 1.0
        chi_z = np.zeros_like(Z);       chi_z[np.abs(Z - zc) < R] = 1.0
        Fx = -0.5 * chi_r * chi_z * (Y - yc)
        Fy =  0.5 * chi_r * chi_z * (X - xc)
        Fz = np.zeros_like(X)  # FIX: array, not scalar
        dFy_dz = np.gradient(Fy, axis=2) / dx
        dFz_dy = np.gradient(Fz, axis=1) / dx
        dFz_dx = np.gradient(Fz, axis=0) / dx
        dFx_dz = np.gradient(Fx, axis=2) / dx
        dFy_dx = np.gradient(Fy, axis=0) / dx
        dFx_dy = np.gradient(Fx, axis=1) / dx
        gx = dFz_dy - dFy_dz
        gy = dFx_dz - dFz_dx
        gz = dFy_dx - dFx_dy
        return A * np.array([gx, gy, gz])

    def spectral_dealias(self, arr_hat):
        for i in range(arr_hat.shape[0]):
            arr_hat[i] *= self.mask
        return arr_hat

    def project_dealias(self, arr):
        ah = np.fft.fftn(arr, axes=(1,2,3))
        ah = self.spectral_dealias(ah)
        return np.fft.ifftn(ah, axes=(1,2,3)).real

    def rhs(self, omega):
        N, L, nu = self.N, self.L, self.nu
        dx = L / N
        omega_hat = np.fft.fftn(omega, axes=(1,2,3))
        omega_hat = self.spectral_dealias(omega_hat)
        u_hat = velocity_from_omega(omega_hat, self.KX, self.KY, self.KZ, self.K2)
        u = np.fft.ifftn(u_hat, axes=(1,2,3)).real
        conv = np.zeros_like(omega)
        for i in range(3):
            conv[i] = u[0]* (np.gradient(omega[i], axis=0)/dx)                     + u[1]* (np.gradient(omega[i], axis=1)/dx)                     + u[2]* (np.gradient(omega[i], axis=2)/dx)
        gu = grad_u(u, L)
        stretch = np.zeros_like(omega)
        for j in range(3):
            stretch[j] = omega[0]*gu[0,j] + omega[1]*gu[1,j] + omega[2]*gu[2,j]
        lap = laplacian_spectral(omega, self.K2)
        return stretch - conv + nu * lap + self.g, u, gu, lap

    def step_RK4(self, w):
        dt = self.dt
        k1, u1, gu1, lap1 = self.rhs(w)
        k2, u2, gu2, lap2 = self.rhs(w + 0.5*dt*k1)
        k3, u3, gu3, lap3 = self.rhs(w + 0.5*dt*k2)
        k4, u4, gu4, lap4 = self.rhs(w + dt*k3)
        w_next = w + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        w_next = self.project_dealias(w_next)
        return w_next, (u4, gu4, lap4)

    def diagnostics(self, w, u, gu, lap):
        L = self.L
        dx3 = (L/self.N)**3
        rho = np.sum(w*w, axis=0)
        integrand = np.zeros_like(rho)
        for i in range(3):
            for j in range(3):
                integrand += rho * w[i] * w[j] * gu[i,j]
        I_stretch = float(np.sum(integrand) * dx3)
        visc_integrand = rho*(w[0]*lap[0] + w[1]*lap[1] + w[2]*lap[2])
        I_visc = float(self.nu * np.sum(visc_integrand) * dx3)
        g = self.g
        I_force = float(np.sum(rho*(w[0]*g[0] + w[1]*g[1] + w[2]*g[2])) * dx3)
        Z = L4norm_of_magnitude(w, L)
        E = L2norm_of_magnitude(w, L)
        return Z, E, I_stretch, I_visc, I_force

def run(params, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'params.json'), 'w') as f:
        json.dump(vars(params), f, indent=2)
    sim = SpectralNS(N=params.N, L=params.L, nu=params.nu, R=params.R, A=params.A,
                     a0=params.a0, dt=params.dt)
    w = sim.omega.copy()
    dt = params.dt
    steps_per_chunk = int(params.T_chunk / dt + 1e-12)
    total_steps = int(params.T_total / dt + 1e-12)
    rows = []
    Z_star = ((2*params.Cnu*params.nu * params.R**(-5/4)) / (params.theta*params.c1))**4
    t = 0.0
    chunk_idx = 0
    next_chunk_end = params.T_chunk
    last_ratio_cross = None
    est_blow_times = []

    for n in range(total_steps+1):
        if n == 0:
            rhs_val, ufields, gufields, lap = sim.rhs(w)
            uN, guN, lapN = ufields, gufields, lap
        Z, E, I_stretch, I_visc, I_force = sim.diagnostics(w, uN, guN, lapN)
        dZdt_est = 4.0*(I_stretch + I_visc + I_force)
        ratio = (I_stretch / (abs(I_visc) + 1e-16)) if (abs(I_visc) > 0) else np.inf
        rows.append([t, Z, E, I_stretch, I_visc, I_force, ratio, dZdt_est])
        if last_ratio_cross is None and ratio > 1.0:
            last_ratio_cross = t
        if Z > 0 and dZdt_est > 0:
            c_inst = dZdt_est / (Z**(5/4))
            if c_inst > 0:
                T_est = t + 4.0/(c_inst * Z**(1/4))
                est_blow_times.append(T_est)
        if n == total_steps:
            break
        w_next, (uN, guN, lapN) = sim.step_RK4(w)
        t += dt
        w = w_next
        if (not np.isfinite(Z)) or np.isnan(Z) or Z > params.Z_hard_cap:
            print('Safety stop at t=', t)
            break
        if t >= next_chunk_end - 1e-12 or n % steps_per_chunk == 0:
            chunk_dir = os.path.join(outdir, f'chunk_{chunk_idx:04d}')
            os.makedirs(chunk_dir, exist_ok=True)
            save_csv(rows, os.path.join(chunk_dir, 'diagnostics_chunk.csv'),
                     header=['t','Z','E','I_stretch','I_visc','I_force','ratio','dZdt_est'])
            np.savez_compressed(os.path.join(chunk_dir, 'state_last.npz'), omega=w)
            rows = []
            chunk_idx += 1
            next_chunk_end += params.T_chunk

    agg_rows = []
    for i in range(chunk_idx):
        chunk_dir = os.path.join(outdir, f'chunk_{i:04d}')
        path = os.path.join(chunk_dir, 'diagnostics_chunk.csv')
        if os.path.exists(path):
            with open(path, 'r') as f:
                rd = csv.reader(f)
                header = next(rd)
                for row in rd:
                    agg_rows.append([float(x) for x in row])
    agg_rows.sort(key=lambda r: r[0])
    save_csv(agg_rows, os.path.join(outdir, 'diagnostics.csv'),
             header=['t','Z','E','I_stretch','I_visc','I_force','ratio','dZdt_est'])

    if len(agg_rows) > 0:
        T = [r[0] for r in agg_rows]
        Zs = [r[1] for r in agg_rows]
        Is = [r[3] for r in agg_rows]
        Iv = [abs(r[4]) for r in agg_rows]
        Ifc = [r[5] for r in agg_rows]
        plt.figure(figsize=(9,5))
        plt.plot(T, Zs, marker='o', linewidth=1.5)
        plt.axhline(Z_star, color='r', linestyle='--', label=f'Analytic Z*={Z_star:.2e}')
        if last_ratio_cross is not None:
            plt.axvline(last_ratio_cross, color='k', linestyle=':', label=f'Ratio>1 at t={last_ratio_cross:.3f}')
        if len(est_blow_times) > 5:
            T_est_med = float(np.median(est_blow_times))
            plt.axvline(T_est_med, color='m', linestyle='-.', label=f'Est. blow-up at t={T_est_med:.3f}')
        plt.title('Z(t) (linear scale)')
        plt.xlabel('t'); plt.ylabel('Z(t)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'Z_linear_with_threshold.png'), dpi=160)
        plt.close()

        plt.figure(figsize=(9,5))
        plt.plot(T, Is, label='I_stretch', marker='o')
        plt.plot(T, Iv, label='|I_visc|', marker='o')
        plt.plot(T, Ifc, label='I_force', marker='o')
        if last_ratio_cross is not None:
            plt.axvline(last_ratio_cross, color='k', linestyle=':', label=f'Ratio>1 at t={last_ratio_cross:.3f}')
        plt.title('Integrals vs t')
        plt.xlabel('t'); plt.ylabel('Integrals')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'integrals_vs_t.png'), dpi=160)
        plt.close()

def build_argparser():
    ap = argparse.ArgumentParser(description='Chunked 3D NS vorticity solver (diagnostics).')
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--L', type=float, default=2*np.pi)
    ap.add_argument('--nu', type=float, default=1e-3)
    ap.add_argument('--R', type=float, default=0.5)
    ap.add_argument('--A', type=float, default=3.0)
    ap.add_argument('--a0', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=2.5e-3)
    ap.add_argument('--T_total', type=float, default=3.0)
    ap.add_argument('--T_chunk', type=float, default=0.5)
    ap.add_argument('--outdir', type=str, default='./ns_out')
    ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--c1', type=float, default=0.348)
    ap.add_argument('--Cnu', type=float, default=3.0)
    ap.add_argument('--Z_hard_cap', type=float, default=1e12)
    return ap

class Params:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def main():
    ap = build_argparser()
    args = ap.parse_args()
    p = Params(**vars(args))
    run(p, p.outdir)

if __name__ == '__main__':
    main()
