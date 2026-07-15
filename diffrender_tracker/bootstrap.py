"""
§5.3 BOOTSTRAP — feature-based global registration (the tracker's cold-start / recovery).

The differentiable renderer can only converge from within ~10-15deg of the answer (brief §6),
so it cannot find the FIRST inter-robot transform on its own, and two independently-moving
robots have no prior on their relative pose. This module supplies that first T with feature
matching instead of rendering:

    downsample -> estimate normals -> FPFH features -> RANSAC global registration -> ICP refine

It is NOT the per-frame tracker (brief §5.3: "do not use this as the per-frame tracker") — it
runs once at startup and on re-init, hands a coarse T to run_fit, and the photometric tracker
takes over. Returns T mapping robot-2 -> robot-1 (same convention as the tracker: merge =
cloud_1 ∪ T·cloud_2), plus a health dict so the caller can reject a bad/overlap-starved fit.

Pure Open3D + numpy — runs on the Mac/CPU, no CUDA.
"""

import numpy as np


def _to_o3d(xyz, rgb):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
    if rgb is not None and len(rgb):
        pc.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(rgb, np.float64), 0, 1))
    return pc


def _preprocess(pc, voxel):
    """Voxel-downsample, estimate normals, compute FPFH features (all sized off `voxel`)."""
    import open3d as o3d
    down = pc.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100))
    return down, fpfh


def bootstrap_register(xyz_a, rgb_a, xyz_b, rgb_b, voxel=0.05, min_fitness=0.15, n_ransac=4):
    """Global-register robot-2's cloud (B) onto robot-1's (A) with no prior.

    Returns (T_2to1 (4,4) float32, info). RANSAC is stochastic, so it's run `n_ransac` times
    and the best-fitness result kept, then refined with COLORED ICP (uses the grayscale
    intensity as well as geometry — helps on the degenerate wall/floor scenes where geometry
    alone slides), falling back to point-to-plane ICP if colored ICP is unavailable/unstable.

    info.success gates on inlier fitness, but note that with genuine PARTIAL overlap fitness is
    inherently capped (only the shared fraction can be an inlier) — the state machine should
    lean on the overlap-coverage health check (§4.4), not this number alone. A good T should
    still land within the tracker's basin so run_fit can refine it."""
    import open3d as o3d
    A, B = _to_o3d(xyz_a, rgb_a), _to_o3d(xyz_b, rgb_b)
    A_d, A_f = _preprocess(A, voxel)
    B_d, B_f = _preprocess(B, voxel)

    dist = voxel * 1.5
    # source=B, target=A  =>  the returned transform maps B -> A  =>  T_2to1.
    best = None
    for _ in range(max(1, n_ransac)):
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            B_d, A_d, B_f, A_f, True, dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.999))
        if best is None or r.fitness > best.fitness:
            best = r

    # refine: colored ICP (intensity + geometry), fall back to point-to-plane on failure.
    try:
        icp = o3d.pipelines.registration.registration_colored_icp(
            B_d, A_d, voxel, best.transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    except Exception:
        icp = o3d.pipelines.registration.registration_icp(
            B_d, A_d, voxel, best.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

    T = np.asarray(icp.transformation, dtype=np.float32)
    info = {
        "ransac_fitness": float(best.fitness),
        "icp_fitness": float(icp.fitness),
        "icp_rmse": float(icp.inlier_rmse),
        "success": bool(icp.fitness >= min_fitness),
    }
    return T, info
