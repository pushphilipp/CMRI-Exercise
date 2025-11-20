"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

from typing import Optional, Tuple

import numpy as np
import utils
from scipy.linalg import fractional_matrix_power, pinv


class Lab05_op:

    def load_data(self, dpath="data_brain_8coils.mat"):
        mat = utils.load_data(dpath)
        kdata = mat["d"]
        sens_maps = mat["c"]
        noise_maps = mat["n"]
        return kdata, sens_maps, noise_maps

    def sos_comb(self, m: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Sum of square.

        :param m: multicoil images [nPE, nRO, nCh]

        :return: mc: combined image [nPE, nRO], dtype: float
        """
        mc = np.sqrt(np.sum(np.abs(m) ** 2, axis=axis))
        return mc

    def ls_comb(
        self, coil_imgs: np.ndarray, sens_maps: np.ndarray, PSI: Optional[np.ndarray] = None, axis: int = -1, **kwargs
    ) -> np.ndarray:
        """
        Least-squares (matched filter)

        :param coil_imgs:               multicoil images [nPE,nRO,nCh]
        :param sens_maps:               coil sensitivity maps [nPE,nRO,nCh]
        :param PSI:                     (Optional) noise correlation matrix [nCh, nCh]

        :return: coil_comb_img:         combined image [nPE,nRO], dtype: complex
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'apply_psi' in this method if you need to used it instead of calling it by self.apply_psi.
        apply_psi = kwargs.get("apply_psi", self.apply_psi)

        coil_imgs_c = coil_imgs
        sens_maps_c = sens_maps

        if PSI is not None:
            # After 1.b.ii.2 and 1.b.ii.3, you should apply the noise correlation matrix to the coil sensitivity maps and coil images.
            sens_maps_c = apply_psi(sens_maps, PSI)
            coil_imgs_c = apply_psi(coil_imgs, PSI)

        numerator = np.sum(np.conj(sens_maps_c) * coil_imgs_c, axis=axis)
        denominator = np.sum(np.abs(sens_maps_c) ** 2, axis=axis)
        coil_comb_img = numerator / denominator
        return coil_comb_img

    def get_psi(self, noise_maps: np.ndarray) -> np.ndarray:
        """
        This function calculates the noise covariance matrix
        Use np.cov function to calculate the covariance matrix.

        Args:
            noise_maps (np.ndarray):    noise maps [nPE,nFE,nCh]

        Returns:
            psi (np.ndarray):           noise covariance matrix [nCh,nCh], dtype: complex
        """
        noise_flat = noise_maps.reshape(-1, noise_maps.shape[-1])
        psi = np.cov(noise_flat, rowvar=False)
        return psi

    def apply_psi(self, x: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        This function applies the coil noise covariance matrix to the image
        \Psi^{-1/2}x
        Use fractional_matrix_power function in the scipy.linalg package for fractional matrix power.
        Use @ operator for matrix multiplication.

        @parak
        x:          matrix of shape [nPE,nRO,nCh]
        psi:        matrix of shape [nCh,nCh]

        @return:
        y:          PSI applied matrix of shape [nPE,nRO,nCh], dtype: complex
        """
        # Raise an error if the dimensions are not correct
        assert (
            x.shape[-1] == psi.shape[0]
        ), "The last dimension of x must be equal to the first dimension of psi"

        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        psi_inv_half = fractional_matrix_power(psi, -0.5)
        y_flat = x_flat @ psi_inv_half
        y = y_flat.reshape(shape)

        return y

    def sense_locs(self, idx_PE: int, PE: int, R: int) -> np.ndarray:
        """
        Get the unwrapped indices for SENSE reconstruction.
        locs are equivalent to u_1, u_2,... in the lecture note.

        :param idx_PE:      index of PE
        :param PE:          number of PE
        :param R:           acceleration factor

        :return: locs:      indices for SENSE reconstruction [R], dtype: int
        """
        step = int(np.ceil(PE / R))
        base = idx_PE % step
        locs = (base + np.arange(R) * step) % PE

        return locs

    def sense_aliased_idx(self, PE: int, R: int, locs: np.ndarray) -> int:
        """
        Get an index for aliased image among indices in locs.
        i_n is an PE index of I_n in the lecture note.

        @param PE:          Length of phase encoding
        @param R:           Acceleration factor
        @param locs:        Indices for SENSE reconstruction at one point [R]

        @return: i_n:       An PE index for aliased images, I_n
        """
        alias_len = int(np.ceil(PE / R))
        i_n = int(locs[0] % alias_len)

        return i_n

    def sense_sm_pinv(self, sens_maps: np.ndarray, locs: np.ndarray, idx_RO: int) -> np.ndarray:
        """
        Get the pseudo-inverse of coil sensitivity maps
        Use pinv function in the scipy.linalg package for puedo inverse of the matrix.

        :param sens_maps:       coil sensitivity maps [nPE,nRO,nCh]
        :param locs:            indices for SENSE reconstruction [R]
        :param idx_RO:          index of RO

        :return: C_pinv:        pseudo-inverse of coil sensitivity maps [R,nCh], dtype: complex
        """
        sens = sens_maps[locs, idx_RO, :].T  # [nCh, R]
        C_pinv = pinv(sens)

        return C_pinv

    def sense_unwrap(self, aliased_imgs: np.ndarray, sm_pinv: np.ndarray, idx_PE_alias: int, idx_RO: int) -> np.ndarray:
        """
        Unwrap the aliased coefficient to get the unaliased image
        Use @ operator for matrix multiplication.

        :param aliased_imgs:        aliased images [Ceil(nPE/R),nRO,nCh]
        :param sm_pinv:             pseudo-inverse of coil sensitivity maps [R,nCh]
        :param idx_PE_alias:        index of PE for aliased image
        :param idx_RO:              index of RO

        :return: unwrapped_coeff:   unwrapped coefficients [R], dtype: complex

        """
        aliased_vec = aliased_imgs[idx_PE_alias, idx_RO, :]
        unwrapped_coeff = sm_pinv @ aliased_vec

        return unwrapped_coeff

    def sense_g_coef(self, sens_maps: np.ndarray, locs: np.ndarray, idx_RO: int) -> np.ndarray:
        """
        Get the g-factor. Use calc_g function in the utils package.

        :param sens_maps:       coil sensitivity maps [nPE,nRO,nCh]
        :param locs:            indices for SENSE reconstruction [R]
        :param idx_RO:          index of RO

        :return: g:             g-factor [R], dtype: float
        """
        sens = sens_maps[locs, idx_RO, :].T  # [nCh, R]
        g = utils.calc_g(sens)

        return g

    def sense_recon(
        self, aliased_imgs: np.ndarray, sens_maps: np.ndarray, PSI: np.ndarray, R: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SENSE reconstruction.

        :param aliased_imgs:        multicoil aliased images [Ceil(nPE/R),nRO,nCh]
        :param sens_maps:           coil sensitivity maps [nPE,nRO,nCh]
        :param PSI:                 noise correlation matrix [nCh, nCh]
        :param R:                   acceleration factor

        :return:
            unaliased_img:          unaliased image [nPE,nRO], dtype: complex
            g_map:                  g-factor map [nPE,nRO], dtype: float
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        apply_psi = kwargs.get("apply_psi", self.apply_psi)
        sense_locs = kwargs.get("sense_locs", self.sense_locs)
        sense_sm_pinv = kwargs.get("sense_sm_pinv", self.sense_sm_pinv)
        sense_g_coef = kwargs.get("sense_g_coef", self.sense_g_coef)
        sense_aliased_idx = kwargs.get("sense_aliased_idx", self.sense_aliased_idx)
        sense_unwrap = kwargs.get("sense_unwrap", self.sense_unwrap)

        PE, RO, _ = sens_maps.shape
        sens_maps = apply_psi(sens_maps, PSI)
        aliased_imgs = apply_psi(aliased_imgs, PSI)

        nonzero_idx = np.array(np.nonzero(np.sum(sens_maps, 2)))
        unaliased_img = np.zeros((PE, RO), dtype=aliased_imgs.dtype)
        g_map = np.zeros((PE, RO))

        for idx_PE, idx_RO in zip(*nonzero_idx):
            if not unaliased_img[idx_PE, idx_RO]:
                # SENSE reconstruction
                locs = sense_locs(idx_PE, PE, R)
                idx_alias = sense_aliased_idx(PE, R, locs)
                sm_pinv = sense_sm_pinv(sens_maps, locs, idx_RO)
                unwrapped = sense_unwrap(aliased_imgs, sm_pinv, idx_alias, idx_RO)
                unaliased_img[locs, idx_RO] = unwrapped

                # g-factor
                g_map[locs, idx_RO] = sense_g_coef(sens_maps, locs, idx_RO)

        return unaliased_img, g_map


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab05 import *

    # %%
    op = Lab05_op()
    kdata, sens_maps, noise_maps = op.load_data()

    # %% Basic coil combinations
    coil_imgs = utils.ifft2c(kdata)
    psi = op.get_psi(noise_maps)

    complex_sum_img = utils.cmplx_sum(coil_imgs)
    sos_img = op.sos_comb(coil_imgs)
    ls_img_no_psi = op.ls_comb(coil_imgs, sens_maps)
    ls_img_psi = op.ls_comb(coil_imgs, sens_maps, psi)

    utils.imshow(
        [complex_sum_img, sos_img, ls_img_no_psi, ls_img_psi],
        titles=["Complex sum", "SoS", "Matched filter\n(no PSI)", "Matched filter\n(with PSI)"],
        suptitle="Coil combination results",
    )

    # %% Cartesian SENSE reconstruction
    R_list = [2, 3, 4]
    psnr_vals = []
    ssim_vals = []

    for R in R_list:
        aliased_kdata = kdata[::R, :, :]
        aliased_imgs = utils.ifft2c(aliased_kdata)

        recon, g_map = op.sense_recon(aliased_imgs, sens_maps, psi, R)
        psnr_vals.append(utils.calc_psnr(np.abs(ls_img_psi), np.abs(recon)))
        ssim_vals.append(utils.calc_ssim(np.abs(ls_img_psi), np.abs(recon)))

        err = utils.normalization(np.abs(recon - ls_img_psi))
        utils.imshow(
            [recon, err, g_map],
            gt=np.abs(ls_img_psi),
            titles=[f"SENSE R={R}", "Reconstruction error", "g-factor"],
            suptitle=f"SENSE reconstruction (R={R})",
            is_mag=False,
        )

    # %% Plot PSNR and SSIM versus acceleration
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(R_list, psnr_vals, marker="o")
    plt.title("PSNR vs Acceleration")
    plt.xlabel("R")
    plt.ylabel("PSNR [dB]")

    plt.subplot(1, 2, 2)
    plt.plot(R_list, ssim_vals, marker="o")
    plt.title("SSIM vs Acceleration")
    plt.xlabel("R")
    plt.ylabel("SSIM")
    plt.tight_layout()
    plt.show()
