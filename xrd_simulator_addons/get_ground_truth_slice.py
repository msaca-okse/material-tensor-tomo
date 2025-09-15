from xrd_simulator.polycrystal import Polycrystal
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm


def get_ground_truth(config = "config.yaml"):


    with open(config, "r") as f:
        config = yaml.safe_load(f)


    session_name = config["session_name"]
    spatial_limit = config["experiment"]["translation_distance"]*config["experiment"]["number_of_translations"]/2

    poly_path = os.path.join("xrd_simulator_addons/samples", f"mesh_{session_name}")
    poly_path = poly_path + '.pc'

    polycrystal = Polycrystal.load(poly_path)


    def _tet_slice_polygon_z0(P, eps=1e-12):
        """
        Intersect a tetra (4x3) with plane z=0.
        Returns:
            None if empty/degenerate, else (K,2) polygon vertices in XY, ordered CCW.
        """
        z = P[:, 2]
        # Quick reject: plane z=0 not between min/max
        if (z.max() < -eps) or (z.min() > eps):
            return None

        # Collect intersection points (XY) from vertices-on-plane and edge crossings
        pts = []

        # vertices on plane
        for v in P:
            if abs(v[2]) <= eps:
                pts.append((v[0], v[1]))

        # edges
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        for i, j in edges:
            zi, zj = P[i,2], P[j,2]
            if zi * zj < -eps**2:  # strictly opposite signs -> proper crossing
                t = zi / (zi - zj)  # z_i + t*(z_j - z_i) = 0
                X = P[i] + t * (P[j] - P[i])
                pts.append((X[0], X[1]))
            # if one endpoint is ~on plane, it's already added above

        if len(pts) < 3:
            return None

        # Deduplicate (robustly)
        arr = np.array(pts)
        rnd = np.round(arr, 12)
        _, uniq_idx = np.unique(rnd, axis=0, return_index=True)
        poly = arr[np.sort(uniq_idx)]

        if poly.shape[0] < 3:
            return None

        # Order CCW around centroid
        c = poly.mean(axis=0)
        ang = np.arctan2(poly[:,1] - c[1], poly[:,0] - c[0])
        poly = poly[np.argsort(ang)]
        return poly  # (K,2)

    def _maybe_zero_based(enod, n_nodes):
        """Ensure 0-based connectivity."""
        if enod.min() == 1 or enod.max() >= n_nodes:
            return enod - 1
        return enod

    def rasterize_mesh_slice(m, classes, xlim, ylim, nx=1024, ny=1024, eps=1e-12, background=-1):
        """
        Slice tetra mesh at z=0 and rasterize element classes (0..3) to an image.

        Args:
        m.coord: (Nnodes,3) float
        m.enod:  (Ne,4) int (0- or 1-based; auto-detected)
        classes: (Ne,) int in {0,1,2,3}
        xlim, ylim: (min,max) domain limits for rasterization
        nx, ny: output resolution
        background: value for pixels outside any sliced element (-1 by default)

        Returns:
        img: (ny,nx) int array
        extent: (xmin, xmax, ymin, ymax) for imshow
        """
        coords = np.asarray(m.coord)
        enod = _maybe_zero_based(np.asarray(m.enod, dtype=int), coords.shape[0])
        classes = np.asarray(classes, dtype=int)
        assert enod.shape[1] == 4
        assert classes.shape[0] == enod.shape[0]

        xmin, xmax = xlim
        ymin, ymax = ylim
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        img = np.full((ny, nx), background, dtype=int)

        # Preselect elements straddling z=0 to avoid unnecessary work
        z_per_node = coords[:, 2]
        zmin = z_per_node[enod].min(axis=1)
        zmax = z_per_node[enod].max(axis=1)
        candidates = np.where((zmin <= eps) & (zmax >= -eps))[0]

        for e in candidates:
            P = coords[enod[e]]  # (4,3)

            # quick XY bbox cull against image domain
            pxmin, pymin = P[:,0].min(), P[:,1].min()
            pxmax, pymax = P[:,0].max(), P[:,1].max()
            if (pxmax < xmin) or (pxmin > xmax) or (pymax < ymin) or (pymin > ymax):
                continue

            poly = _tet_slice_polygon_z0(P, eps=eps)
            if poly is None:
                continue  # no area at z=0

            # Polygon bbox -> pixel index ranges
            bx0 = max(0, int(np.floor((poly[:,0].min() - xmin) / dx)))
            bx1 = min(nx - 1, int(np.floor((poly[:,0].max() - xmin) / dx)))
            by0 = max(0, int(np.floor((poly[:,1].min() - ymin) / dy)))
            by1 = min(ny - 1, int(np.floor((poly[:,1].max() - ymin) / dy)))
            if bx0 > bx1 or by0 > by1:
                continue

            ix = np.arange(bx0, bx1 + 1)
            iy = np.arange(by0, by1 + 1)

            # Pixel centers in that window
            x_cent = xmin + (ix + 0.5) * dx
            y_cent = ymin + (iy + 0.5) * dy
            XX, YY = np.meshgrid(x_cent, y_cent)
            pts = np.column_stack([XX.ravel(), YY.ravel()])

            mask = Path(poly).contains_points(pts)
            mask = mask.reshape(iy.size, ix.size)

            yy, xx = np.where(mask)
            if yy.size:
                img[iy[yy], ix[xx]] = classes[e]

        extent = (xmin, xmax, ymin, ymax)
        return img[::-1], extent
        #return np.transpose(img, [1, 0])[::1, ::-1], extent




    emap = polycrystal.element_phase_map
    olab = polycrystal.orientation_lab
    crystal_mask = np.array([1 if x is not None else 0 for x in olab], dtype=np.int8)
    coords = polycrystal.mesh_lab.coord
    enod = polycrystal.mesh_lab.enod.shape
    m = polycrystal.mesh_lab


    #--- Example usage (assumes you have m.coord, m.enod, and `classes`) ---
    xlim = (-spatial_limit, spatial_limit)
    ylim = (-spatial_limit, spatial_limit)
    phase, extent = rasterize_mesh_slice(m, emap, xlim, ylim, nx=1024, ny=1024, background=-1)
    crystal_map, extent = rasterize_mesh_slice(m, crystal_mask, xlim, ylim, nx=1024, ny=1024, background=-1)
    crystal_map = crystal_map == 1
    return phase, crystal_map