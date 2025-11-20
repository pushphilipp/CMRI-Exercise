"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester
"""

from typing import Tuple

import numpy as np
import torch
import torchkbnufft as tkbn
import utils
from grid import grid


class Lab04_op:

    def __init__(self, device_num=0):
        self.PI = np.pi
        self.GA = 111.246117975 * self.PI / 180  # Golden angle

        # For NUFFT operator
        self.device = torch.device(f"cuda:{device_num}") if torch.cuda.is_available() else torch.device("cpu")

    def load_kdata(self):
        mat = utils.load_data("radial_data.mat")
        k_radial = mat["k"]
        return k_radial

    def get_traj(self, k_radial) -> np.complex128:
        """
        This method returns the trajectory of the radial k-space data.

        Args:
            k_radial:           k-space data. (shape of [Readout, Spokes])

        Return:
            traj:               trajectory of the k-space data. (shape of [Readout])
        """
        readout, spokes = k_radial.shape

        radii = np.linspace(-0.5, 0.5, readout)[:, None]
        angles = self.PI / 2 + self.GA * np.arange(spokes)[None, :]
        traj = radii * np.exp(1j * angles)

        return traj

    def calc_nyquist(self, k_radial) -> int:
        """
        This method calculates the minimum number of spokes that satisfy Nyquist sampling theorm.

        Args:
            k_radial:        k-space data. (shape of [Readout, Spokes])

        Returns:
            spokes_nyq:      minimum number of spokes that satisfy Nyquist sampling theorem. (Ceilling up integer)
        """
        readout, _ = k_radial.shape
        spokes_nyq = int(np.ceil(self.PI * readout / 2))

        return spokes_nyq

    def grid_radial(self, k_radial, traj) -> np.complex128:
        """
        This method maps the radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])

        Returns:
            k_cart:         Cartesian k-space data. (shape of [Readout, Readout])
        """
        readout, _ = k_radial.shape
        k_cart = grid(k_radial, traj, readout)

        return k_cart

    def get_ramp(self, k_radial) -> np.float64:
        """
        This method returns a ramp filter that can be used to weight the kspace data
        for gridding.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            ramp:           ramp filter (shape of [Readout, 1])
        """
        readout, _ = k_radial.shape
        radii = np.linspace(-0.5, 0.5, readout)
        ramp = np.abs(radii)[:, None]

        return ramp

    def grid_radial_ds(self, k_radial, traj, **kwargs) -> np.complex128:
        """
        This method maps the density compensatied radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])

        Returns:
            k_cart_ds:      Densitiy compensated Cartesian k-space data. (shape of [Readout, Readout])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        ramp = get_ramp(k_radial)
        k_radial_ds = k_radial * ramp
        readout, _ = k_radial.shape
        k_cart_ds = grid(k_radial_ds, traj, readout)

        return k_cart_ds

    def grid_radial_ds_os(self, k_radial, traj, os_rate: int, **kwargs) -> np.complex128:
        """
        This method maps the density compensatied radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout, Spokes])
            os_rate:        oversampling rate

        Returns:
            k_cart_ds_os:   Densitiy compensated oversampled Cartesian k-space data. (shape of [Readout * os_rate, Readout * os_rate])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        ramp = get_ramp(k_radial)
        k_radial_ds = k_radial * ramp
        readout, _ = k_radial.shape
        k_cart_ds_os = grid(k_radial_ds, traj, readout * os_rate)

        return k_cart_ds_os

    def center_crop_2d(self, ov_image, target_shape: Tuple[int, int]) -> np.complex128:
        """
        This method crops the image to the desired size.

        Args:
            ov_image (np.ndarray):          Oversmapled image (shape of [Oversampled_Readout, Oversampled_Readout])
            target_shape (tuple):           Desired shape of the image

        Returns:
            image_crop (np.ndarray):        cropped image (shape of target_shape)
        """
        target_h, target_w = target_shape
        h, w = ov_image.shape

        start_y = (h - target_h) // 2
        start_x = (w - target_w) // 2

        image_crop = ov_image[start_y:start_y + target_h, start_x:start_x + target_w]

        return image_crop

    def decimate2d(self, ov_image, os_rate: int) -> np.complex128:
        """Downsample an oversampled image by cropping to the central region."""

        target_shape = (ov_image.shape[0] // os_rate, ov_image.shape[1] // os_rate)
        return self.center_crop_2d(ov_image, target_shape)

    def get_c(self, k_radial, traj, os_rate) -> np.complex128:
        """
        This method calculates the deapodization factor c(x,y) for the deapodization.
        Think about how to utilize the grid function to calculate c(x,y).
        If you want to use delta function, set 1 of the delta function at the center of the k-space. (ex. at Readout//2)

        Args:
            k_radial (np.ndarray):      k-space data. (shape of [Readout, Spokes])
            traj (np..ndarray):         trajectory of the k-space data. (shape of [Readout, Spokes])
            os_rate (int):              oversampling rate

        Returns:
            c:                          deapodization factor c(x,y) (shape of [Oversampled_Readout, Oversampled_Readout])
        """
        readout, spokes = k_radial.shape
        delta = np.zeros_like(k_radial, dtype=np.complex128)
        delta[readout // 2, :] = 1.0

        kernel_k = grid(delta, traj, readout * os_rate)
        c = utils.ifft2c(kernel_k)

        return c

    def deapodization(self, k_radial, traj, os_rate, **kwargs) -> np.complex128:
        """
        This method performs deapodization on the k-space data.
            1. Get the gridding kernel using get_c method.
            2. Grid the k-space data with opversampling and density compensation.
            3. Reconstruction the grid k-space data.
            4. Deapodize the reconstructed k-space data.


        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])
            os_rate:        oversampling rate

        Returns:
            deapod_recon:   De-apodized reconstruction image (shape of [Readout, Readout])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'grid_radial_ds_os' and 'decimate2d' in this method if you need to used them instead of calling them by self.grid_radial_ds_os and self.decimate2d.
        grid_radial_ds_os = kwargs.get("grid_radial_ds_os", self.grid_radial_ds_os)
        center_crop_2d = kwargs.get("center_crop_2d", self.center_crop_2d)
        get_c = kwargs.get("get_c", self.get_c)

        a = 1

        c = get_c(k_radial, traj, os_rate)
        k_cart_ds_os = grid_radial_ds_os(k_radial, traj, os_rate)
        recon_os = utils.ifft2c(k_cart_ds_os)

        eps = 1e-8
        deapod = recon_os / (c + a + eps)

        deapod_recon = center_crop_2d(deapod, (k_radial.shape[0], k_radial.shape[0]))

        return deapod_recon

    def nufft_traj(self, k_radial) -> torch.Tensor:
        """
        This method returns the trajectory of the radial k-space data for NUFFT.
        - The lenght of the trajectory spock for NUFFT operator is 2 * PI (-PI ~ PI) and the first spock starts at the angle of 90 degrees.
        - Trajectory for the NUFFT operator needs to be in shape of (2, x) where 2 is the separate channels of real and imaginary part of the trajectory.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            ktraj:          trajectory of the k-space data for NUFFT. (shape of [2, Readout * Spokes])
        """
        readout, spokes = k_radial.shape
        radii = np.linspace(-self.PI, self.PI, readout)[:, None]
        angles = self.PI / 2 + self.GA * np.arange(spokes)[None, :]
        traj = radii * np.exp(1j * angles)
        ktraj_np = np.vstack([traj.real.ravel(order="F"), traj.imag.ravel(order="F")])
        ktraj = torch.tensor(ktraj_np, dtype=torch.float32, device=self.device)

        return ktraj

    def nufft_kdata(self, k_radial, **kwargs) -> torch.Tensor:
        """
        This method prepares the k-space data for NUFFT.
        - Use density commpenstated k-space data for NUFFT.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            nufft_kdata:    NUFFT k-space data. (shape of [Batch(1), Coil(1), Readout*Spokes])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        ramp = get_ramp(k_radial)
        k_ds = k_radial * ramp
        k_ds = k_ds.ravel(order="F")
        nufft_kdata = torch.tensor(k_ds, dtype=torch.complex64, device=self.device)
        nufft_kdata = nufft_kdata.unsqueeze(0).unsqueeze(0)

        return nufft_kdata

    def nufft_recon(self, k_radial, im_size: Tuple[int, int], **kwargs) -> np.ndarray:
        """
        This method reconstructs the image from the k-space data using NUFFT.
        - Define trajectories and k-data for a NUFFT operator.
        - Define a Adjoint NUFFT operator with im_size. Do not forget to set the self.device.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            im_size:        size of the desired reconstruction image.

        Returns:
            nufft_recon:    NUFFT reconstructed image. (shape of im_size)
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'nufft_traj' and 'nufft_kdata' in this method if you need to used them instead of calling them by self.nufft_traj and self.nufft_kdata.
        nufft_traj = kwargs.get("nufft_traj", self.nufft_traj)
        nufft_kdata = kwargs.get("nufft_kdata", self.nufft_kdata)

        ktraj = nufft_traj(k_radial)
        kdata = nufft_kdata(k_radial)

        adj_op = tkbn.KbNufftAdjoint(im_size=im_size).to(self.device)
        recon = adj_op(kdata, ktraj)
        recon = recon.squeeze().cpu().numpy()
        nufft_recon = recon

        return nufft_recon


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab04 import *

    # %%
    op = Lab04_op()
    k_radial = op.load_kdata()
    utils.imshow([k_radial], norm=0.3, suptitle="Raw radial k-space")

    traj = op.get_traj(k_radial)
    utils.plot_spokes(traj, 10)

    spokes_nyq = op.calc_nyquist(k_radial)
    print(f"Nyquist spokes (min): {spokes_nyq}")

    k_cart = op.grid_radial(k_radial, traj)
    recon = utils.ifft2c(k_cart)
    utils.imshow([np.log1p(np.abs(k_cart))], suptitle="Triangular gridded k-space (log)" )
    utils.imshow([recon], suptitle="Basic gridding reconstruction", norm=0.4)

    k_cart_ds = op.grid_radial_ds(k_radial, traj)
    recon_ds = utils.ifft2c(k_cart_ds)
    utils.imshow([np.log1p(np.abs(k_cart_ds))], suptitle="Density compensated gridded k-space (log)")
    utils.imshow([recon_ds], suptitle="Density compensated reconstruction", norm=0.4)

    os_rate = 2
    k_cart_ds_os = op.grid_radial_ds_os(k_radial, traj, os_rate)
    recon_ds_os = utils.ifft2c(k_cart_ds_os)
    recon_ds_os_crop = op.center_crop_2d(recon_ds_os, k_radial.shape)
    utils.imshow(
        [np.log1p(np.abs(k_cart_ds_os))],
        suptitle=f"Oversampled ({os_rate}x) gridded k-space (log)",
    )
    utils.imshow(
        [recon_ds_os, recon_ds_os_crop],
        titles=["Oversampled recon", "Cropped recon"],
        suptitle="Oversampled reconstructions",
        norm=0.4,
    )

    deapod_recon = op.deapodization(k_radial, traj, os_rate)
    utils.imshow([deapod_recon], suptitle="Deapodized reconstruction", norm=0.4)

    nufft_recon = op.nufft_recon(k_radial, (k_radial.shape[0], k_radial.shape[0]))
    utils.imshow([nufft_recon], suptitle="NUFFT adjoint reconstruction", norm=0.4)
