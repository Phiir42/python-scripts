import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

try:
    from lmfit import Minimizer, Parameters
except ImportError as e:
    raise ImportError("Please `pip install lmfit` to enable peak fitting.") from e


def _cars_model_complex(x, par):
    """Complex-sum CARS model; returns |sum(...)|^2"""
    x = np.asarray(x, dtype=float)

    def _p(name, default=0.0):
        v = par[name] if (hasattr(par, "__getitem__") and name in par) else default
        return getattr(v, "value", v)

    cc = np.zeros_like(x, dtype=np.complex128)
    for k in range(1, 8):
        A = _p(f"A{k}")
        x0 = _p(f"x{k}")
        w = _p(f"w{k}")
        cc += A / ((x - x0) + 1j * w)

    cc += _p("ANR", 0.0)
    for tag in ("OR1", "OR2"):
        A = _p(f"A{tag}", 0.0)
        x0 = _p(f"x{tag}", 0.0)
        w = _p(f"w{tag}", 20.0)
        cc += A / ((x - x0) + 1j * w)
    return np.abs(cc) ** 2


def _residual(par, x, y):
    return _cars_model_complex(x, par) - y


def _make_default_params_from_config(config):
    """(Unchanged docstring and logic from original; seeds/bounds from config['peak_fit'])."""
    pf = config.get("peak_fit", {})
    centers_seed = pf.get(
        "centers_seed", [2859.9, 2884.7, 2936.0, 3019.7, 2812.0, 2908.6, 2967.7]
    )
    centers_low = pf.get("centers_low", [2845, 2875, 2925, 3012, 2805, 2900, 2950])
    centers_high = pf.get("centers_high", [2860, 2895, 2940, 3022, 2825, 2915, 3000])
    amps_seed = pf.get("amps_seed", [13.92, 8.84, 7.08, 0.28, 1.25, 9.77, -1.41])
    amp_min = pf.get("amp_min", None)
    amp_max = pf.get("amp_max", None)
    widths_seed = pf.get("widths_seed", [10, 21.6, 14.8, 10.18, 20, 12, 22])
    widths_low = pf.get("widths_low", [5, 5, 5, 8, 5, 5, 5])
    widths_high = pf.get("widths_high", [30, 30, 30, 30, 30, 30, 30])

    p = Parameters()
    for i in range(7):
        p.add(
            f"x{i+1}",
            value=centers_seed[i],
            min=centers_low[i],
            max=centers_high[i],
            vary=True,
        )
        p.add(f"A{i+1}", value=amps_seed[i], min=amp_min, max=amp_max, vary=True)
        p.add(
            f"w{i+1}",
            value=widths_seed[i],
            min=widths_low[i],
            max=widths_high[i],
            vary=True,
        )

    p.add(
        "ANR",
        value=pf.get("ANR_seed", 0.02),
        min=pf.get("ANR_min", 0),
        vary=pf.get("ANR_vary", True),
    )
    p.add(
        "xOR1",
        value=pf.get("xOR1_seed", 3549.0),
        min=pf.get("xOR1_min", 3260),
        max=pf.get("xOR1_max", 4000),
        vary=pf.get("xOR1_vary", True),
    )
    p.add(
        "AOR1",
        value=pf.get("AOR1_seed", 100.0),
        min=pf.get("AOR1_min", 0),
        vary=pf.get("AOR1_vary", True),
    )
    p.add(
        "wOR1",
        value=pf.get("wOR1_seed", 20.0),
        min=pf.get("wOR1_min", 5),
        max=pf.get("wOR1_max", 60),
        vary=pf.get("wOR1_vary", True),
    )
    p.add(
        "xOR2",
        value=pf.get("xOR2_seed", 0.0),
        min=pf.get("xOR2_min", 0),
        max=pf.get("xOR2_max", 1500),
        vary=pf.get("xOR2_vary", False),
    )
    p.add("AOR2", value=pf.get("AOR2_seed", 0.0), vary=pf.get("AOR2_vary", False))
    p.add(
        "wOR2",
        value=pf.get("wOR2_seed", 20.0),
        min=pf.get("wOR2_min", 5),
        max=pf.get("wOR2_max", 60),
        vary=pf.get("wOR2_vary", False),
    )
    return p


def _seed_centers_from_data(
    x, y, target_centers=(2855, 2890, 2935, 3019, 2812, 2909, 2968)
):
    # smooth a little for peak finding
    from skimage.filters import gaussian  # <-- fix: import the submodule directly

    y_s = gaussian(y, sigma=1, preserve_range=True)
    pk_idx, _ = find_peaks(y_s, prominence=np.percentile(y_s, 60) / 6.0, distance=2)
    pk_x = x[pk_idx]
    centers = []
    for t in target_centers:
        centers.append(
            float(t if len(pk_x) == 0 else pk_x[np.argmin(np.abs(pk_x - t))])
        )
    return centers


def fit_cars_peaks(wavenumbers_cm1, intensity, config):
    """(Unchanged docstring and strategies A/B/C from original.)"""
    x = np.asarray(wavenumbers_cm1, dtype=float)
    y = np.asarray(intensity, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - np.min(y)
    ymax = np.max(y)
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    y = y / ymax

    def _run(params, label):
        minner = Minimizer(_residual, params, fcn_args=(x, y))
        res = minner.least_squares(
            method="trf", max_nfev=4000, loss="soft_l1", f_scale=0.1
        )
        diag = {
            "success": bool(res.success),
            "redchi": getattr(res, "redchi", np.nan),
            "nfev": getattr(res, "nfev", np.nan),
            "message": getattr(res, "message", ""),
            "active_bounds": list(getattr(res, "active_mask", [])),
            "strategy_used": label,
        }
        out = {f"x{k}": res.params[f"x{k}"].value for k in range(1, 8)}
        out.update({f"A{k}": res.params[f"A{k}"].value for k in range(1, 8)})
        out.update(
            {
                "ANR": (
                    res.params.get("ANR", None).value if "ANR" in res.params else 0.0
                ),
                "xOR1": (
                    res.params.get("xOR1", None).value if "xOR1" in res.params else 0.0
                ),
                "AOR1": (
                    res.params.get("AOR1", None).value if "AOR1" in res.params else 0.0
                ),
                "wOR1": (
                    res.params.get("wOR1", None).value if "wOR1" in res.params else 20.0
                ),
                "xOR2": (
                    res.params.get("xOR2", None).value if "xOR2" in res.params else 0.0
                ),
                "AOR2": (
                    res.params.get("AOR2", None).value if "AOR2" in res.params else 0.0
                ),
                "wOR2": (
                    res.params.get("wOR2", None).value if "wOR2" in res.params else 20.0
                ),
            }
        )
        out.update(diag)
        # also record widths for accurate reconstruction
        out.update({f"w{k}": res.params[f"w{k}"].value for k in range(1, 8)})

        # normalized predicted curve at the original x (same space as y given to fit)
        try:
            y_hat = _cars_model_complex(x, res.params)  # returns |sum|^2 already
        except Exception:
            y_hat = None
        out["y_fit"] = y_hat  # normalized units; caller will rescale for plotting

        return out

    pA = _make_default_params_from_config(config)
    outA = _run(pA, "default-seeds")
    if outA["success"] and np.isfinite(outA["redchi"]) and outA["redchi"] < 1e-2:
        return outA

    pB = _make_default_params_from_config(config)
    new_centers = _seed_centers_from_data(x, y)
    for i, c in enumerate(new_centers, 1):
        lo = pB[f"x{i}"].min if pB[f"x{i}"].min is not None else (c - 20)
        hi = pB[f"x{i}"].max if pB[f"x{i}"].max is not None else (c + 20)
        pB[f"x{i}"].set(value=float(np.clip(c, lo, hi)))
    outB = _run(pB, "data-seeded-centers")
    if outB["success"] and (
        not np.isfinite(outA["redchi"]) or outB["redchi"] <= outA["redchi"]
    ):
        return outB

    pC = _make_default_params_from_config(config)
    for i in range(1, 8):
        pC[f"w{i}"].set(vary=False, value=np.clip(pC[f"w{i}"].value, 8.0, 24.0))
    for i in (4, 7):
        pC[f"A{i}"].set(value=0.0, min=0.0)
    outC = _run(pC, "fixed-widths/reduced")
    return outC


def _plot_peak_fit_debug(
    x_cm1, y_vals, fit_result, droplet_id, category, location, marker
):
    """Plot raw hyperspectrum, 7 individual peak components, and the total fit."""
    import matplotlib.pyplot as plt
    import numpy as np

    # raw x/y
    x_cm1 = np.asarray(x_cm1, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    # scale used during fitting (if the spectrum was normalized beforehand)
    scale = float(fit_result.get("_scale", 1.0))

    # dense grid for smooth plotting
    x_dense = np.linspace(float(np.min(x_cm1)), float(np.max(x_cm1)), 400)
    try:
        y_smooth = make_interp_spline(x_cm1, y_vals, k=3)(x_dense)
    except Exception:
        y_smooth = np.interp(x_dense, x_cm1, y_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(x_dense, y_smooth, "-", color="black", lw=1.5, label="Raw (spline)")
    plt.scatter(x_cm1, y_vals, c="k", s=25, marker="o", label="Raw points")

    if fit_result.get("success", False):
        # 1) Total fit (prefer model-provided y_fit in normalized units)
        y_fit_norm = fit_result.get("y_fit", None)
        if y_fit_norm is not None:
            y_fit_norm = np.asarray(y_fit_norm, dtype=float)
            try:
                y_fit_dense = make_interp_spline(x_cm1, y_fit_norm, k=3)(x_dense)
            except Exception:
                y_fit_dense = np.interp(x_dense, x_cm1, y_fit_norm)
            plt.plot(x_dense, y_fit_dense * scale, "r-", lw=2, label="Total fit")
        else:
            # reconstruct if y_fit wasn't saved
            cc_total = np.zeros_like(x_dense, dtype=np.complex128)
            for k in range(1, 8):
                Ak = float(fit_result.get(f"A{k}", 0.0))
                x0 = float(fit_result.get(f"x{k}", 0.0))
                wk = float(fit_result.get(f"w{k}", 20.0))
                if wk != 0:
                    cc_total += Ak / ((x_dense - x0) + 1j * wk)
            cc_total += float(fit_result.get("ANR", 0.0))
            for tag in ("OR1", "OR2"):
                Ak = float(fit_result.get(f"A{tag}", 0.0))
                x0 = float(fit_result.get(f"x{tag}", 0.0))
                wk = float(fit_result.get(f"w{tag}", 20.0))
                if wk != 0:
                    cc_total += Ak / ((x_dense - x0) + 1j * wk)
            y_fit = np.abs(cc_total) ** 2
            plt.plot(x_dense, y_fit * scale, "r-", lw=2, label="Total fit")

        # 2) Individual peak components (A1..A7, x1..x7, w1..w7)
        colors = plt.cm.tab10.colors
        for k in range(1, 8):
            Ak = float(fit_result.get(f"A{k}", 0.0))
            x0 = float(fit_result.get(f"x{k}", 0.0))
            wk = float(fit_result.get(f"w{k}", 20.0))
            if Ak == 0.0 or wk == 0.0:
                continue
            comp = np.abs(Ak / ((x_dense - x0) + 1j * wk)) ** 2
            plt.plot(
                x_dense,
                comp * scale,
                "--",
                lw=1.2,
                color=colors[(k - 1) % len(colors)],
                label=f"Peak {k} ({x0:.0f} cm⁻¹)",
            )

        # (optional off-resonant components — uncomment if you want to visualize them too)
        # for tag, idx in (("OR1", 8), ("OR2", 9)):
        #     Ak = float(fit_result.get(f"A{tag}", 0.0))
        #     x0 = float(fit_result.get(f"x{tag}", 0.0))
        #     wk = float(fit_result.get(f"w{tag}", 20.0))
        #     if Ak != 0.0 and wk != 0.0:
        #         comp = np.abs(Ak / ((x_dense - x0) + 1j * wk)) ** 2
        #         plt.plot(x_dense, comp * scale, ':', lw=1.0,
        #                  color=colors[idx % len(colors)],
        #                  label=f"{tag} ({x0:.0f} cm⁻¹)")

    # Title/labels
    ttl = f"LipidID {droplet_id} | {category}, {location}, {marker}"
    if "strategy_used" in fit_result:
        try:
            ttl += f"  [{fit_result['strategy_used']}, χ²≈{fit_result.get('redchi', np.nan):.3g}]"
        except Exception:
            ttl += f"  [{fit_result['strategy_used']}]"
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(ttl)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
