# odf_sh_image_operator.py
from typing import Optional, Sequence
import numpy as np

from cil.framework import ImageGeometry, ImageData
from cil.optimisation.operators import LinearOperator

from spherical_harmonics import SphericalHarmonics
from odf_geometry import ODFGeometry


class ODFSHOperator3d(LinearOperator):
    """
    B: coefficients -> ODF-detector values

    Domain  (ig_in):  ImageGeometry with shape (K, N, H, W)
                      channels=K (SH coeffs), z=N (projections), y=H, x=W
    Range   (ig_out): ImageGeometry with shape (M, N, H, W)
                      channels=M (detector segments), z=N, y=H, x=W

    Notes
    -----
    - Only `direct()` is implemented (and returns ImageData).
    - `adjoint()` is left as a skeleton (raises NotImplementedError).
    - Internally, SH expects last axis to be K and first to be N when
      evaluating all projections; we rearrange axes accordingly.
    """

    def __init__(
        self,
        odf_geometry: ODFGeometry,
        ig_in: ImageGeometry,
        ig_out: ImageGeometry,
        indices: Optional[Sequence[int]] = None,
        dtype=np.float32,
        **sh_kwargs
    ):
        self.geom    = odf_geometry
        self.ig_in   = ig_in
        self.ig_out  = ig_out
        self.indices = None if indices is None else np.asarray(indices, dtype=int)
        self.dtype   = dtype

        # Build SH on the probed directions

        self.geom.build_probed_coordinates()
        pc = self.geom.probed_coordinates

        self.sh = SphericalHarmonics(
            probed_coordinates=pc,
            ell_max=self.geom.ell_max,
            enforce_friedel_symmetry=self.geom.enforce_friedel_symmetry,
            **sh_kwargs
        )

        # Shapes from SH
        N_sh, M_sh, K_sh = map(int, self.sh.projection_matrix.shape)
        self.N, self.M, self.K = N_sh, M_sh, K_sh

        # Validate geometry compatibility
        self._validate_geometries()

        # Initialise LinearOperator with geometries (what CIL expects)
        super().__init__(domain_geometry=self.ig_in, range_geometry=self.ig_out)

    # ------------ LinearOperator API ------------

    def direct(self, x: ImageData) -> ImageData:
        """
        x: ImageData on ig_in with array shape (K, N, H, W)
        returns ImageData on ig_out with array shape (M, N, H, W)
        """
        xin = x.as_array()

        if xin.dtype != self.dtype:
            xin = xin.astype(self.dtype, copy=False)

        if xin.shape != self.ig_in.shape:
            raise ValueError(f"Input shape {xin.shape} != ig_in.shape {self.ig_in.shape}")

        x_sh = np.transpose(xin, (1, 2, 3, 0))  # (N, H, W, K)
        Nx, Ny, Nz, K = np.shape(x_sh)
        x_sh_flat = x_sh.reshape(-1,K)


        # Call SH forward
        y_sh_flat = self.sh.forward(x_sh_flat, indices=self.indices).astype(self.dtype, copy=False) # the input has dim (K, N), the output should have dim (M,N)
        M = y_sh_flat.shape[-1]
        y_out_arr = y_sh_flat.reshape(Nx, Ny, Nz, M)
        y_out_arr = y_out_arr.transpose([3,0,1,2])
        # y_sh shape: (N, H, W, M)


        if tuple(y_out_arr.shape) != tuple(self.ig_out.shape):
            raise RuntimeError(
                f"Computed output shape {y_out_arr.shape} doesn't match ig_out.shape {self.ig_out.shape}."
            )

        out = self.ig_out.allocate(0, dtype=self.dtype)
        out.fill(y_out_arr)
        return out

    def adjoint(self, y: ImageData):
        """
        x: ImageData on ig_in with array shape (K, N, H, W)
        returns ImageData on ig_out with array shape (M, N, H, W)
        """
        yin = y.as_array()

        if yin.dtype != self.dtype:
            yin = yin.astype(self.dtype, copy=False)

        if yin.shape != self.ig_out.shape:
            raise ValueError(f"Input shape {yin.shape} != ig_in.shape {self.ig_out.shape}")

        y_sh = np.transpose(yin, (1, 2, 3, 0))  # (N, H, W, K)
        Nx, Ny, Nz, M = np.shape(y_sh)
        y_sh_flat = y_sh.reshape(-1,M)


        # Call SH forward
        x_sh_flat = self.sh.adjoint(y_sh_flat, indices=self.indices).astype(self.dtype, copy=False) # the input has dim (K, N), the output should have dim (M,N)
        K = x_sh_flat.shape[-1]
        x_out_arr = x_sh_flat.reshape(Nx, Ny, Nz, K)
        x_out_arr = x_out_arr.transpose([3,0,1,2])
        # y_sh shape: (N, H, W, K)


        if tuple(x_out_arr.shape) != tuple(self.ig_in.shape):
            raise RuntimeError(
                f"Computed output shape {x_out_arr.shape} doesn't match ig_out.shape {self.ig_in.shape}."
            )

        out = self.ig_in.allocate(0, dtype=self.dtype)
        out.fill(x_out_arr)
        return out

    # ------------ helpers ------------

    def _validate_geometries(self):
            # ig_in: (K, 1, H, W)
            if self.ig_in.channels != self.K:
                raise ValueError(f"ig_in.channels={self.ig_in.channels} must equal K={self.K}")
            V_in, H_in, W_in =self.ig_in.shape[1], self.ig_in.shape[2], self.ig_in.shape[3]

            # ig_out: (M, N, H, W)
            if self.ig_out.channels != self.M:
                raise ValueError(f"ig_out.channels={self.ig_out.channels} must equal M={self.M}")
            V_out, H_out, W_out = self.ig_out.shape[1], self.ig_out.shape[2], self.ig_out.shape[3]
            if (V_in, H_in, W_in) != (V_out, H_out, W_out):
                raise ValueError(f"Spatial (H,W) mismatch: ig_in {(V_in, H_in,W_in)} vs ig_out {(V_out, H_out,W_out)}")







class ODFSHOperator2d(LinearOperator):
    """
    B: coefficients -> ODF-detector values

    Domain  (ig_in):  ImageGeometry with shape (K, N, H)
                      channels=K (SH coeffs), z=N (projections), y=H
    Range   (ig_out): ImageGeometry with shape (M, N, H)
                      channels=M (detector segments), z=N, y=H

    Notes
    -----
    - Only `direct()` is implemented (and returns ImageData).
    - `adjoint()` is left as a skeleton (raises NotImplementedError).
    - Internally, SH expects last axis to be K and first to be N when
      evaluating all projections; we rearrange axes accordingly.
    """

    def __init__(
        self,
        odf_geometry: ODFGeometry,
        ig_in: ImageGeometry,
        ig_out: ImageGeometry,
        indices: Optional[Sequence[int]] = None,
        dtype=np.float32,
        **sh_kwargs
    ):
        self.geom    = odf_geometry
        self.ig_in   = ig_in
        self.ig_out  = ig_out
        self.indices = None if indices is None else np.asarray(indices, dtype=int)
        self.dtype   = dtype

        # Build SH on the probed directions

        self.geom.build_probed_coordinates()
        pc = self.geom.probed_coordinates

        self.sh = SphericalHarmonics(
            probed_coordinates=pc,
            ell_max=self.geom.ell_max,
            enforce_friedel_symmetry=self.geom.enforce_friedel_symmetry,
            **sh_kwargs
        )

        # Shapes from SH
        N_sh, M_sh, K_sh = map(int, self.sh.projection_matrix.shape)
        self.N, self.M, self.K = N_sh, M_sh, K_sh

        # Validate geometry compatibility
        self._validate_geometries()

        # Initialise LinearOperator with geometries (what CIL expects)
        super().__init__(domain_geometry=self.ig_in, range_geometry=self.ig_out)

    # ------------ LinearOperator API ------------

    def direct(self, x: ImageData) -> ImageData:
        """
        x: ImageData on ig_in with array shape (K, N, H)
        returns ImageData on ig_out with array shape (M, N, H)
        """
        xin = x.as_array()

        if xin.dtype != self.dtype:
            xin = xin.astype(self.dtype, copy=False)

        if xin.shape != self.ig_in.shape:
            raise ValueError(f"Input shape {xin.shape} != ig_in.shape {self.ig_in.shape}")

        x_sh = np.transpose(xin, (1, 2, 0))  # (N, H, K)
        Nx, Ny, K = np.shape(x_sh)
        x_sh_flat = x_sh.reshape(-1,K)


        # Call SH forward
        y_sh_flat = self.sh.forward(x_sh_flat, indices=self.indices).astype(self.dtype, copy=False) # the input has dim (K, N), the output should have dim (M,N)
        M = y_sh_flat.shape[-1]
        y_out_arr = y_sh_flat.reshape(Nx, Ny, M)
        y_out_arr = y_out_arr.transpose([2,0,1])
        # y_sh shape: (N, H, M)


        if tuple(y_out_arr.shape) != tuple(self.ig_out.shape):
            raise RuntimeError(
                f"Computed output shape {y_out_arr.shape} doesn't match ig_out.shape {self.ig_out.shape}."
            )

        out = self.ig_out.allocate(0, dtype=self.dtype)
        out.fill(y_out_arr)
        return out

    def adjoint(self, y: ImageData):
        """
        x: ImageData on ig_in with array shape (K, N, H)
        returns ImageData on ig_out with array shape (M, N, H)
        """
        yin = y.as_array()

        if yin.dtype != self.dtype:
            yin = yin.astype(self.dtype, copy=False)

        if yin.shape != self.ig_out.shape:
            raise ValueError(f"Input shape {yin.shape} != ig_in.shape {self.ig_out.shape}")

        y_sh = np.transpose(yin, (1, 2, 0))  # (N, H, M)
        Nx, Ny, M = np.shape(y_sh)
        y_sh_flat = y_sh.reshape(-1,M)


        # Call SH forward
        x_sh_flat = self.sh.adjoint(y_sh_flat, indices=self.indices).astype(self.dtype, copy=False) # the input has dim (K, N), the output should have dim (M,N)
        K = x_sh_flat.shape[-1]
        x_out_arr = x_sh_flat.reshape(Nx, Ny, K)
        x_out_arr = x_out_arr.transpose([2,0,1])
        # y_sh shape: (N, H, K)


        if tuple(x_out_arr.shape) != tuple(self.ig_in.shape):
            raise RuntimeError(
                f"Computed output shape {x_out_arr.shape} doesn't match ig_out.shape {self.ig_in.shape}."
            )

        out = self.ig_in.allocate(0, dtype=self.dtype)
        out.fill(x_out_arr)
        return out

    # ------------ helpers ------------

    def _validate_geometries(self):
            # ig_in: (K, 1, H)
            if self.ig_in.channels != self.K:
                raise ValueError(f"ig_in.channels={self.ig_in.channels} must equal K={self.K}")
            V_in, H_in =self.ig_in.shape[1], self.ig_in.shape[2]

            # ig_out: (M, N, H)
            if self.ig_out.channels != self.M:
                raise ValueError(f"ig_out.channels={self.ig_out.channels} must equal M={self.M}")
            V_out, H_out = self.ig_out.shape[1], self.ig_out.shape[2]
            if (V_in, H_in) != (V_out, H_out):
                raise ValueError(f"Spatial (H,W) mismatch: ig_in {(V_in, H_in)} vs ig_out {(V_out, H_out)}")
            




class ODFSHOperator2d_fast(LinearOperator):
    """
    B: coefficients -> ODF-detector values

    Domain  (ig_in):  ImageGeometry with shape (Nx, Ny, K)
                      z=K (SH coeffs)
    Range   (ig_out): ImageGeometry with shape (Nx, Ny, M)
                      z=M (detector segments)

    Notes
    -----
    - Only `direct()` is implemented (and returns ImageData).
    - `adjoint()` is left as a skeleton (raises NotImplementedError).
    - The operator treats the z-dimension as if it were the channel axis.
    """

    def __init__(
        self,
        odf_geometry: ODFGeometry,
        ig_in: ImageGeometry,
        ig_out: ImageGeometry,
        indices: Optional[Sequence[int]] = None,
        dtype=np.float32,
        **sh_kwargs
    ):
        self.geom    = odf_geometry
        self.ig_in   = ig_in
        self.ig_out  = ig_out
        self.indices = None if indices is None else np.asarray(indices, dtype=int)
        self.dtype   = dtype

        # Build SH on the probed directions
        self.geom.build_probed_coordinates()
        pc = self.geom.probed_coordinates

        self.sh = SphericalHarmonics(
            probed_coordinates=pc,
            ell_max=self.geom.ell_max,
            enforce_friedel_symmetry=self.geom.enforce_friedel_symmetry,
            **sh_kwargs
        )

        # Shapes from SH projection matrix
        N_sh, M_sh, K_sh = map(int, self.sh.projection_matrix.shape)
        self.N, self.M, self.K = N_sh, M_sh, K_sh

        # Validate geometry compatibility
        self._validate_geometries()

        # Initialise LinearOperator with geometries
        super().__init__(domain_geometry=self.ig_in, range_geometry=self.ig_out)

    # ------------ LinearOperator API ------------

    def direct(self, x: ImageData) -> ImageData:
        """
        x: ImageData on ig_in with array shape (Nx, Ny, K)
        returns ImageData on ig_out with array shape (Nx, Ny, M)
        """
        xin = x.as_array()
        K, Nx, Ny = np.shape(xin)

        if xin.dtype != self.dtype:
            xin = xin.astype(self.dtype, copy=False)

        if tuple(xin.shape) != tuple(self.ig_in.shape):
            raise ValueError(f"Input shape {xin.shape} != ig_in.shape {self.ig_in.shape}")

        
        x_sh_flat = xin.reshape(K, Nx*Ny).transpose([1,0])  # (Nx*Ny, K)

        # Call SH forward: (Nx*Ny, K) -> (Nx*Ny, M)
        y_sh_flat = self.sh.forward(x_sh_flat, indices=self.indices).astype(self.dtype, copy=False)

        y_out_arr = y_sh_flat.transpose([1,0]).reshape(self.M, Nx, Ny)  # (Nx, Ny, M)

        if tuple(y_out_arr.shape) != tuple(self.ig_out.shape):
            raise RuntimeError(
                f"Computed output shape {y_out_arr.shape} doesn't match ig_out.shape {self.ig_out.shape}."
            )

        out = self.ig_out.allocate(0, dtype=self.dtype)
        out.fill(y_out_arr)
        return out

    def adjoint(self, y: ImageData):
        """
        x: ImageData on ig_in with array shape (Nx, Ny, K)
        returns ImageData on ig_out with array shape (Nx, Ny, M)
        """
        yin = y.as_array()
        M, Nx, Ny = np.shape(yin)

        if yin.dtype != self.dtype:
            yin = yin.astype(self.dtype, copy=False)

        if tuple(yin.shape) != tuple(self.ig_out.shape):
            raise ValueError(f"Input shape {yin.shape} != ig_in.shape {self.ig_out.shape}")

        
        y_sh_flat = yin.reshape(M, Nx*Ny).transpose([1,0])  # (Nx*Ny, K)

        x_sh_flat = self.sh.adjoint(y_sh_flat, indices=self.indices).astype(self.dtype, copy=False)

        x_out_arr = x_sh_flat.transpose([1,0]).reshape(self.K, Nx, Ny)  # (Nx, Ny, M)

        if tuple(x_out_arr.shape) != tuple(self.ig_in.shape):
            raise RuntimeError(
                f"Computed output shape {x_out_arr.shape} doesn't match ig_out.shape {self.ig_in.shape}."
            )

        out = self.ig_in.allocate(0, dtype=self.dtype)
        out.fill(x_out_arr)
        return out
    

    def _validate_geometries(self):
        # ig_in: (Nx, Ny, K)
        if self.ig_in.shape[0] != self.K:
            raise ValueError(f"ig_in.shape[-1]={self.ig_in.shape[-1]} must equal K={self.K}")

        # ig_out: (Nx, Ny, M)
        if self.ig_out.shape[0] != self.M:
            raise ValueError(f"ig_out.shape[-1]={self.ig_out.shape[-1]} must equal M={self.M}")

        if self.ig_in.shape[1:] != self.ig_out.shape[1:]:
            raise ValueError(f"Spatial (x,y) mismatch: ig_in {self.ig_in.shape[1:]} vs ig_out {self.ig_out.shape[1:]}")
