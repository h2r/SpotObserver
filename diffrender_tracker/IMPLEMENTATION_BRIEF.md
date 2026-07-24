# Implementation Brief — Real-Time Multi-Spot Point-Cloud Merging via Differentiable Photometric Pose Tracking

**Audience:** the coding agent implementing this. You have no prior context; this document is complete.
**Status:** the core algorithm is already prototyped and empirically validated on synthetic data (see §3). Your job is to turn that prototype into a real-time tracker running on real robot streams (§5).

---

## 0. TL;DR

Two Boston Dynamics Spot robots stream colored point clouds to an off-board machine (RTX 3080). We continuously estimate the 6-DoF rigid transform **T** that aligns robot 2's cloud to robot 1's frame, and publish the merged cloud in real time. **T is estimated by differentiable rendering**: render both clouds from a virtual camera, minimize a masked *color* (photometric) loss in image space, backprop to the pose. Each frame is **warm-started** from the previous frame's pose and takes only ~8 gradient steps, which is what makes it real-time.

The differentiable tracker is only the **TRACKING** mode. A complete tracker is a state machine: **BOOTSTRAP** (feature-based global registration to find the first T, since the renderer cannot cold-start) → **TRACKING** (warm-started renderer) → re-init on overlap loss.

---

## 1. Goal & constraints

- **Goal:** merge point clouds from 2 (later N) independently-moving Spot robots into a common frame, in real time.
- **Compute:** off-board RTX 3080. GPU is not the bottleneck; iteration count is (solved by warm-starting). VRAM is a non-issue at this scale.
- **Clouds:** ~20k points each after onboard voxel-downsampling, with **per-point RGB** (color is essential — see §2).
- **Reference frame:** robot 1's current frame. **Both robots move**, so the merged output rides with robot 1 (see §7.3).
- **Latency:** drop-to-latest always — process the newest cloud, never a backlog — to keep the per-frame motion small.

---

## 2. Decisions already made — DO NOT relitigate

These were decided and, where noted, **empirically validated**. Do not substitute your own approach for these without a very good reason.

1. **Color/photometric loss, NOT geometric (ICP/Chamfer/depth).** The target scenes are geometrically degenerate (flat walls, corridors, planar floors) where a geometry-only objective slides freely along the unconstrained direction. **Validated (§6):** with a depth loss the pose converged in rotation but translation error blew up to 152 cm; with the color loss it reached 0.3 cm. Color is doing work no geometric loss can. **Do not use ICP or a depth-based objective as the optimizer.**
2. **Differentiable renderer = Gaussian splatting.** A hard z-buffer point renderer has zero/undefined gradient w.r.t. pose. Splatting spreads each point into a soft blob so the image (and loss) is a smooth function of the pose. The prototype uses a pure-torch soft splatter; production swaps in `gsplat`.
3. **Isotropic, point-derived Gaussians.** One scalar radius per point, identity rotation. Only the Gaussian **means** depend on the pose — keep the autograd path short: `se(3) twist → exp map → transform means → render → loss`.
4. **Warm-started tracking, not cold solving.** The loss is non-convex; a cold solve needs 100+ iterations. Real-time comes from warm-starting each frame from the previous pose and taking ~8 steps. **Validated (§6):** 8 warm-started steps/frame track 51° of accumulated drift at 0.8° mean error; 8 cold steps/frame fail (23° error).
5. **The renderer cannot cold-start.** It only converges from within ~11–15° of the answer (§6). Therefore a separate **feature-based global registration** provides the first T and re-init — this is required, not optional.
6. **Color to optimize, depth to validate.** Depth is useless as the objective but valuable as a *consistency check* to catch wrong-but-low-loss poses (§4.3).

---

## 3. The validated prototype (your starting point)

Located in `diffrender_tracker/`. All pure-torch, runs on CPU/MPS (no CUDA), validated on a synthetic pair. **Bring this directory to the target branch and build on it.** Keep the interfaces below stable — production changes swap *implementations* behind them.

| File | Purpose | Key interface |
|---|---|---|
| `gen_synthetic_pair.py` | Generates a synthetic colored cloud pair (textured wall+floor corner, partial overlap, known ground-truth T, baked-in exposure gap) → `data/cloud_A.npz`, `data/cloud_B.npz`. Keep as a regression fixture. | writes `xyz (N,3) f32`, `rgb (N,3) f32 [0,1]`; B also has `R_gt,t_gt,T_gt` |
| `gaussians.py` | Cloud → isotropic Gaussians. | `cloud_to_gaussians(xyz, rgb, scale=None, scale_mult=1.5, opacity=0.99, device) -> {means,quats,scales,opacities,colors,spacing}` |
| `camera.py` | Virtual pinhole camera, **OpenCV convention** (x right, y down, z forward), gsplat-compatible. | `VirtualCamera.look_at(...)`, `.place_overlap(means_a,means_b,...)`, `.scaled(factor)`, `.project(points)->(u,v,z,valid)`; fields `K(3,3)`, `viewmat(4,4) world→cam`, `width`, `height` |
| `render.py` | **Differentiable render** (torch soft splatter). Swap target for gsplat. | `render(gaussians, camera, transform=None) -> {image(H,W,3), alpha(H,W), depth(H,W)}` — `transform` is a (4,4) applied to means only; gradients flow through it |
| `se3.py` | se(3) exp map + pose error. | `se3_exp(xi6)->T(4,4)`, `pose_error(T_est,T_gt)->(deg,m)` |
| `loss.py` | Masked photometric loss (optimizer) + depth loss (validator). | `photometric_loss(moving,target,tau=1e-3,huber_delta=0.1,affine=False)->(loss,info)`; `depth_loss(...)` |
| `tracker_core.py` | Reusable warm-startable fit. | `run_fit(gA,gB,T_gt,cam,T_init,pyramid,lr,affine,loss_mode,xi_init)->(T_final,hist)` |
| `fit_pose.py` | Cold-solve convergence test (coarse-to-fine). Use as regression test. | — |
| `tracker_realtime.py` | **The tracking loop** — warm-started 8-steps/frame over a simulated trajectory. This is the loop that goes real-time; replace the simulated trajectory with the real stream. | `solve_frame(gB_k,cam,target,T_init,steps=8,lr=0.03)` |
| `ablation_depth_vs_color.py`, `basin_sweep.py` | Experiments that produced the findings in §6. Keep for reference. | — |

**Conventions to preserve:**
- **T maps robot-2 frame → robot-1 frame.** Merge = `cloud_1 ∪ (T · cloud_2)`.
- Pose is parameterized as `T = T_init @ se3_exp(xi)` with `xi` a 6-vector `[translation(3), rotation(3)]` starting at 0 → the optimizer moves a *delta* from the warm-start seed.
- Camera is OpenCV convention so `K`/`viewmat` drop straight into gsplat.

---

## 4. Production architecture

### 4.1 Runtime data flow
```
ROBOT 1 (reference)                 ROBOT 2 (moving)
 depth+RGB → colored cloud           depth+RGB → colored cloud
 voxel-downsample ~20k               voxel-downsample ~20k
 compress → stream ─────┐            compress → stream ─────┐
                        ▼                                   ▼
             OFF-BOARD (RTX 3080), drop-to-latest
     cloud_1 → target render (re-render on update)   cloud_2 → Gaussians
                          └────────► TRACKER ◄────────┘  estimates T (2→1)
                                        ▼
                        merged = cloud_1 ∪ (T · cloud_2) → publish
```

### 4.2 State machine
```
        ┌────────────┐   T valid & healthy    ┌──────────────┐
 ──────►│ BOOTSTRAP  │───────────────────────►│   TRACKING   │
        │ global reg │                        │ warm 8-step  │
        └────────────┘◄───────────────────────└──────────────┘
              ▲   overlap collapsed / diverged / jump (no-overlap signal)
              └────────────────────────────────────┘
```
- **BOOTSTRAP**: feature-based global registration (§5.3) → initial T with no prior. Validate; on success → TRACKING.
- **TRACKING**: warm-started differentiable-render solve (§5.1–5.2), ~8 steps/frame.

### 4.3 Per-frame TRACKING step
```
1. Ingest newest cloud_2  → Gaussians
2. If cloud_1 changed      → re-render target (§7.3)
3. Predict: T_pred = T_prev · velocity            (constant-velocity / IMU; §5.5)
4. Solve:   ~8 gradient steps from T_pred → T_new  (the prototype solver)
5. Health check (§4.4). Healthy? yes → keep. no → emit no-overlap signal → BOOTSTRAP
6. Merge & publish: cloud_1 ∪ (T_new · cloud_2)
7. Update velocity: velocity = T_prev⁻¹ · T_new
```

### 4.4 Health checks — MUST NOT rely on loss magnitude
A confidently-wrong pose can have a *low* loss (validated: the depth-degenerate case and the texture-alias failure both have low loss). Triggers, all loss-independent:
- **Overlap coverage** = fraction of mutually-covered pixels (`alpha_1 & alpha_2`). Below threshold → no overlap → re-init. (This is the "no-overlap signal.")
- **Temporal-jump guard** (cheapest, most important for a tracker): a warm-started solve should move slowly. Any single-frame jump `> motion budget (~5–6° / ~8 cm, §6)` is almost certainly a texture alias → reject, hold/predict; if it persists a few frames → re-init.
- **3D geometric validator** (optional but recommended): after solving, compute the point-to-plane (or Chamfer) residual between `cloud_1` and `T·cloud_2` in 3D. This is where depth earns its keep — it catches aliases the color loss is blind to.

---

## 5. Components to build (ordered, with acceptance criteria)

### 5.1 Swap the renderer to gsplat  *(do first)*
Replace the soft-splatter internals of `render()` with `gsplat.rasterization`, keeping the **exact same signature and return dict**.
- Map the gaussians dict → gsplat args: `means(N,3)`, `quats(N,4)`, `scales(N,3)`, `opacities(N,)`, `colors(N,3)`, `viewmats = camera.viewmat[None]` (world→cam), `Ks = camera.K[None]`, `width`, `height`. Use `render_mode="RGB+D"` to get depth; derive `alpha` from the returned alphas.
- Apply `transform` to `means` before rasterizing (same as prototype). Gradients must flow through `transform`.
- **Acceptance:** on `data/cloud_A.npz`/`cloud_B.npz`, `fit_pose.py` (retargeted to the gsplat renderer) still converges from ~9°/14 cm to <0.5°/1 cm. gsplat render of A visually matches the soft-splatter render. Per-frame solve time measured (target: well under real-time budget; expect ~1–3 ms/render).

### 5.2 Streaming ingest from PySpotObserver  *(do second)*
Replace the simulated trajectory in `tracker_realtime.py` with real clouds.
- Inspect `PySpotObserver/` (start at `PySpotObserver/examples/basic_streaming.py` and the `pyspotobserver` module) for how to pull clouds off each robot.
- Produce **colored** clouds (per-point RGB) — Spot depth registered to an RGB camera. **RISK:** if the current pipeline yields geometry-only clouds, adding RGB registration is prerequisite work; flag early. Color is non-negotiable (§2).
- Voxel-downsample to ~20k onboard/pre-wire; **drop-to-latest** queue (process newest, discard backlog).
- **Acceptance:** the TRACKING loop runs on two live/recorded Spot streams and publishes a T and a merged cloud each frame at the newest-frame rate, from a hand-provided or bootstrap initial T.

### 5.3 BOOTSTRAP: feature-based global registration  *(required)*
The renderer's first T and its recovery path. Suggested: Open3D `registration_ransac_based_on_feature_matching` on FPFH features (voxel-downsample → estimate normals → FPFH → RANSAC), then a short refinement (colored-ICP is fine here — this is bootstrap, not the tracking objective). **Do not** use this as the per-frame tracker; it's the cold-start/recovery only.
- **Acceptance:** from two clouds with unknown relative pose (no prior), returns a T within the tracker's basin (~<10°/15 cm) that then converges under TRACKING. Runs when overlap is sufficient; reports failure otherwise.

### 5.4 Health checks & state-machine wiring  *(required)*
Implement §4.4 and the §4.2 transitions. Emit an explicit no-overlap signal on re-init.
- **Acceptance:** injecting a fast motion / removing overlap flips the tracker to BOOTSTRAP and it recovers; a synthetic single-frame texture-alias jump is rejected by the jump guard, not accepted.

### 5.5 Motion prediction  *(polish)*
Seed step 3 with a constant-velocity (optionally IMU) prediction instead of the previous pose. Cancels the steady-state warm-start lag observed in the prototype.
- **Acceptance:** per-frame steady-state error under constant robot motion is lower with prediction than without.

### 5.6 Merge & publish  *(trivial)*
`merged = concat(cloud_1, T · cloud_2)`; publish merged cloud + T on whatever transport the rest of the system uses. Poses streamed back to the robots if needed.

---

## 6. Key parameters & empirical findings (from the validated prototype)

- **Convergence:** cold solve 9.3°/14 cm → 0.06°/0.3 cm (coarse-to-fine, ~230 iters), *with* exposure gap and high-frequency texture active.
- **Basin of attraction:** converges up to **~11–15° / ~17–22 cm**, then falls out. → **Motion budget for warm-start tracking: keep inter-frame relative motion under ~5–6° / ~8 cm** (half the basin, for margin). Fire re-init above this.
- **Failure mode:** past the basin the optimizer snaps to a **repeated-texture alias** at a consistent wrong offset **with LOW loss** → hence §4.4's loss-independent health checks. Repetitive scenes (tiled floors, warehouse racking, uniform corridors) are the main risk; coarser pyramid levels help disambiguate.
- **Warm-start:** 8 steps/frame tracks; 8 cold steps/frame fails. Coarse-to-fine pyramid (e.g. `[(0.25,·),(0.5,·),(1.0,·)]`) widens the basin — used for cold solve / bootstrap refinement; a single full-res scale is fine for the small per-frame warm-start delta.
- **Exposure gap:** two robots = two auto-exposures. `photometric_loss(..., affine=True)` fits a per-channel gain+bias to cancel it. Keep it on for real cross-robot data. (Alternative if affine is insufficient: switch the loss to NCC / image-gradient / census, which are exposure-invariant.)
- **Loss:** masked Huber on **color** over the overlap (`alpha_1 & alpha_2 > tau`) only.

---

## 7. Gotchas / non-obvious constraints

1. **gsplat is CUDA-only.** It will not run on a Mac/CPU dev box. The pure-torch soft splatter in `render.py` is the CPU/MPS stand-in behind the same interface — keep it as a fallback/test path so the pipeline is debuggable without CUDA.
2. **Two independently-moving robots have NO fixed extrinsic.** The relative pose changes whenever either robot moves, so it cannot be calibrated once — it must be estimated every frame (that's the whole point). Kalibr does NOT apply to the inter-robot transform (it calibrates a rigid single-robot rig; use it only for per-robot camera intrinsics/extrinsics if needed — Spot SDK intrinsics suffice to start).
3. **Both robots move → the cached-target trick partially breaks.** The prototype caches the reference render once (assumes robot 1 static). In production, **re-render the target whenever a new cloud_1 arrives** (cheap on gsplat). The merged output then rides in robot 1's *current* frame — fine for a relative merge. For a fixed world frame, anchor each robot with its own odometry/SLAM and let the tracker correct inter-robot drift (later concern; start robot-1-relative).
4. **Virtual camera placement matters for a color loss.** Place the virtual camera so it views the textured overlap surface roughly frontally (so the color gradient constrains the in-plane DoF). `VirtualCamera.place_overlap` auto-places at the overlap centroid; revisit if scenes change.
5. **Keep the `render()` interface stable.** Everything downstream (loss, tracker, tests) depends on `{image, alpha, depth}`. Swap implementations, not signatures.

---

## 8. Environment & dependencies

- **Dev/test (CPU/MPS):** `numpy`, `torch`, `open3d` (viz + bootstrap), `matplotlib`. The prototype and all regression tests run here without CUDA.
- **Production (3080):** add `gsplat` (CUDA). Everything else identical.
- **Regression gate:** port `gen_synthetic_pair.py` + `fit_pose.py` to the target branch and confirm convergence still passes after the gsplat swap — this is the cheapest guard that the CUDA path matches validated behavior.

---

## 9. Suggested milestone order
1. Port `diffrender_tracker/` to the target branch; confirm `fit_pose.py` converges (soft splatter). *(baseline)*
2. §5.1 gsplat swap + regression gate. *(real frame rate)*
3. §5.2 streaming ingest (+ RGB clouds if missing). *(real data through TRACKING)*
4. §5.3 bootstrap global registration. *(first-frame T + recovery)*
5. §5.4 health checks + state machine. *(robustness)*
6. §5.5 motion prediction, §5.6 publish. *(polish + output)*
