"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import random
from pathlib import Path
from typing import List, NamedTuple, Tuple

import h5py
import numpy as np
import torch
import utils
from fastmri_loss import SSIMLoss
from fastmri_unet import NormUnet
from fastmri_utils import (center_crop_to_smallest, complex_abs, complex_conj,
                           complex_mul, fft2c, ifft2c, to_tensor)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True

# Pretrained weights of UNet and VarNet
PRET_WEIGHTS_UNET = utils.get_save_path("exercise") / "UNet_weights_pretrained.pkl"
PRET_WEIGHTS_VARNET = utils.get_save_path("exercise") / "VarNet_weights_pretrained.pkl"
DATA_PATH = Path("/home/woody/mrvl/mrvl101h/CMRI/Lab10")

class Lab10_op:

    def __init__(self, data_root=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = DATA_PATH if data_root is None else data_root  # This path should be fixed!
        self.save_dir = utils.get_save_path("exercise")
        seed_everything()

    @staticmethod
    def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        It expands a single coil image to multiple coil image by multiplying with sensitivity maps.
        Then it returns the corresponding multi-coil kspace
        (Hint: to multiple the image with sensitivity maps, you can use complex_mul function)

        Args:
            x:              Single coil image [Batch, 1, RO, PE, 2]
            sens_maps:      Sensitivity maps [Batch, Coil, RO, PE, 2]

        Returns:
            torch.Tensor:   Multi-coil kspace [Batch, Coil, RO, PE, 2]
        """
        # Your code here ...

        return None

    @staticmethod
    def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        It reduce multi-coil kspace to a single coil image by multiplying with complex conjugate of sensitivity maps then summing up along the coil dimention. Keep in mind that the shape should not be changed after coil combination. (see the option, 'keepdim' in torch.sum function)
        (Hint: to multiple the image with sensitivity maps, you can use complex_mul function)

        Args:
            x:              Multi-coil kspace [Batch, Coil, RO, PE, 2]
            sens_maps:      Sensitivity maps [Batch, Coil, RO, PE, 2]

        Returns:
            torch.Tensor:   Single coil image [Batch, 1, RO, PE, 2]
        """
        # Your code here ...

        return None

    def trainer(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Adam,
        criterion: nn.Module,
        epochs=25,
        early_stop: int = 5,
        checkpoint_path: str = None,
        verbose=True,
        model_name: str = "VarNet",
    ) -> Tuple[List[float], List[float]]:
        """
        Train a given model with the supplied data.
        To deal with different size of images, please look at "center_crop_to_smallest()."

        Args:
            model:                      Neural network.
            train_dataloader:           DataLoader yielding the training data.
            validation_dataloader:      DataLoader yielding the validation data.
            optimizer:                  Optimizer for training.
            criterion:                  Loss function.
            epochs:                     Total training epochs.
            early_stop:                 Number of epochs to wait for improvement before stopping.
            checkpoint_path:            Path to a file containing network weights.
            verbose:                    Show progress during training.
            model_name:                 Name of the model.

        Returns:
            recorded train and validation loss (training_loss, validation_loss)
        """
        # setup
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        weight_path = self.save_dir / f"{model_name}_weights.pkl"

        # If checkpoint_path is given, load the saved state_dict to the model. Do not forget to map parameters on the same device when you load the state_dict.
        # Your code here ...

        # Define early_stop_counter
        # Your code here ...

        # training_loss and validation_loss.
        training_loss, validation_loss = (
            (list(), list()) if checkpoint_path is None else utils.load_loss(self.save_dir, model_name)
        )

        # model training
        best_validation_loss = np.inf
        with tqdm(desc=f"Epoch {1:4d}", total=epochs, unit="epoch", disable=not verbose) as pbar:
            for epoch in range(1, epochs + 1):
                # Reset running_training_loss and running_validation_loss
                # Your code here ...

                # training step
                # Your code here ...

                # Loop over the training dataset
                # Your code here ...
                # Forward pass
                # Your code here ...

                # Compute the loss.
                # To match the size of the output and target, you can use center_crop_to_smallest function.
                # Look at the batch sample for max_value
                # Your code here ...

                # Zero the gradients
                # Your code here ...

                # Backward pass
                # Your code here ...

                # Update the weights
                # Your code here ...

                # Accumulate the training loss
                # Your code here ...

                # validation step
                # Your code here ...

                # Loop over the validation dataset
                # Your code here ...
                # Forward pass for validation. Keep in mind that you should not update the weights during validation.
                # Your code here ...

                # Compute the validation loss.
                # Your code here ...

                # Accumulate the validation loss
                # Your code here ...

                # recording intermediate results
                # Your code here ...

                # Save the model if the validation loss of this step is the best validation loss.
                # If you save the model, reset the early_stop_counter to 0, otherwise, increase the early_stop_counter by 1.
                # Your code here ...

                # Early stopping check
                # Your code here ...

                # Update the progress bar.
                pbar.set_description(f"Epoch {epoch:4d}")
                pbar.set_postfix(
                    {
                        "training loss": f"{training_loss[-1]:.5f}",
                        "validation loss": f"{validation_loss[-1]:.5f}",
                        "best": f"{best_validation_loss:.5f}",
                    }
                )
                pbar.update()

        return training_loss, validation_loss

    def tester(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        checkpoint_path: str = None,
        model_name: str = "VarNet",
        verbose=True,
    ):
        """
        Evaluate a given model with the supplied data.
        The test results are stored in the SAVE_DIR/Results folder.
        To deal with different size of images, please look at "center_crop_to_smallest()."

        Args:
            model:                      Neural network.
            test_dataloader:            DataLoader yielding the test data.
            checkpoint_path:            Path to a file containing network weights.
            model_name:                 Name of the model.
            verbose:                    Show progress during training.
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / f"{model_name}_weights.pkl"

        RECON_DIR = (
            self.save_dir / "Pretrained" if "pretrained" in checkpoint_path.stem else self.save_dir / "Finetuned"
        )

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()

        psnr, ssim = [], []
        with tqdm(
            desc=f"Case {0:5d}", total=len(test_dataloader), unit="case", leave=True, disable=not verbose
        ) as pbar:
            for case, batch in enumerate(test_dataloader, start=1):
                with torch.no_grad():
                    output = model(batch.masked_kspace, batch.sens_maps, batch.mask)

                target, output = center_crop_to_smallest(batch.target, output)

                target = target.cpu().numpy()
                output = output.cpu().numpy()

                for i in range(output.shape[0]):
                    utils.imshow(
                        [output[i]],
                        gt=target[i],
                        titles=["Output"],
                        suptitle=batch.filename[i],
                        filename=f"{model_name}_{batch.filename[i]}",
                        root=RECON_DIR,
                    )
                    np.save(RECON_DIR / "Results" / f"{model_name}_{batch.filename[i]}", output[i])
                    np.save(RECON_DIR / "Results" / f"target_{batch.filename[i]}", target[i])
                    psnr.append(utils.calc_psnr(target[i], output[i]))
                    ssim.append(utils.calc_ssim(target[i], output[i]))

                pbar.set_description(f"Case {case:5d}")
                pbar.update()

        utils.pprint(f"- Average metrics for all test data", level=1)
        utils.pprint(f"- PSNR: {np.mean(psnr):.2f} dB", level=2)
        utils.pprint(f"- SSIM: {np.mean(ssim)*100:.2f} %", level=2)

    def get_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        Get the model based on the model name. [VarNet, UNet]
        Transfer the model to the self.device.
        """
        def count_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Your code here ...
        model = None

        utils.pprint(f"- Model: {model_name}", level=1)
        utils.pprint(f"- Trainable parameters: {count_trainable_params(model)}", level=1)
        return model

    def get_loss(self):
        """
        Define the loss function.

        Returns:
            loss (nn.Module):  Loss function
        """
        # Your code here ...
        return None

    def get_optimizer(self, model, lr=0.001):
        """
        Define the optimizer.

        Args:
            model (nn.Module):  Neural network model
            lr (float):         Learning rate

        Returns:
            optimizer (torch.optim.Optimizer):  Optimizer
        """
        # Your code here ...
        return None


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace:      K-space after applying sampling mask. [nCoil, Readout, Phase encoding, 2]
        target:             Target image. [Cropped readout, Cropped phase encoding], e.g., [320, 320]
        mask:               Applied sampling mask. [1, 1, Phase encoding, 1]
        sens_maps:          Sensitivity maps [nCoil, Readout, Phase encoding, 2]
        fname:              File name.
        max_value:          Maximum image value.
    """

    masked_kspace: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    sens_maps: torch.Tensor
    filename: str
    max_value: float


class FastMRIDataset(Dataset):
    """
    A PyTorch Dataset that provides access to MR images.
    """

    def __init__(self, root: Path, prototype: bool = False):
        """
        Args:
            root: Path to the dataset.
            prototype: Whether to use only the first 5 volumes.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples = list(root.iterdir())
        self.samples = samples if not prototype else samples[:5]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        """
        Read the i-th sample and return it as a VarNetSample.
        """
        # Your code here ...

        sample = None

        return sample


class UNet(nn.Module):
    """
    U-Net architecture

    Args:
        chans: Number of channels.
        pools: Number of encoder and decoder parts.

    self:
        model: U-Net model.

    Methods:
        forward(masked_kspace, sens_maps, ignore): Forward pass of the network.
            Args:
                - masked_kspace: Input k-space with the applied mask. [Batch, Coil, RO, PE, 2]
                - sens_maps: Sensitivity maps. [Batch, Coil, RO, PE, 2]
                - ignore: Ignored parameter. (This is for compatibility with the VarNet class.)

            Returns:
                - torch.Tensor: Absolute-valued reconstructed image. [Batch, RO, PE]
                                Hint: To take the absolute value of complex numbers, you can use complex_abs function.
    """

    def __init__(self, chans: int = 18, pools: int = 2, **kwargs):
        """
        Args:
            chans: Number of channels.
            pools: Number of encoder and decoder parts.
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        sens_reduce = kwargs.get("sens_reduce", Lab10_op.sens_reduce)
        self.sens_reduce = sens_reduce

        super().__init__()

        # Your code here ...

    def forward(self, masked_kspace: torch.Tensor, sens_maps: torch.Tensor, ignore=None) -> torch.Tensor:
        # Your code here ...
        return None


class VarNetBlock(nn.Module):
    """
    Model block for variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.

    Args:
        model: Module for "regularization" component of variational network.

    self:
        model: Model for the regularization component.
        dc_weight: Weight for the soft data consistency term. The dc_weight should be trainable.

    Methods:
        forward(u, ref_kspace, mask, sens_maps): Forward pass of the network.
            Args:
                - u: Input image. [Batch, 1, RO, PE, 2]
                - ref_kspace: Reference k-space. [Batch, Coil, RO, PE, 2]
                - mask: Applied sampling mask. [Batch, 1, 1, PE, 1]
                - sens_maps: Sensitivity maps. [Batch, Coil, RO, PE, 2]

            Returns:
                - torch.Tensor: Intermediate reconstruction image. [Batch, 1, RO, PE, 2]
    """

    def __init__(self, model: nn.Module, **kwargs):
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        sens_reduce = kwargs.get("sens_reduce", Lab10_op.sens_reduce)
        sens_expand = kwargs.get("sens_expand", Lab10_op.sens_expand)
        self.sens_reduce = sens_reduce
        self.sens_expand = sens_expand

        super().__init__()

        # Your code here ...

    def forward(
        self, u: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:

        # Your code here ...

        return None


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net regularizer.

    Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            chans: Number of channels for cascade U-Net.
            pools: Number of encoder and decoder parts for cascade U-Net.

    self:
        cascades: List of VarNetBlocks.

    Methods:
        forward(masked_kspace, sens_maps, mask): Forward pass of the network.
            Args:
                - masked_kspace: Input k-space with the applied mask. [Batch, Coil, RO, PE, 2]
                - sens_maps: Sensitivity maps. [Batch, Coil, RO, PE, 2]
                - mask: Applied sampling mask. [Batch, 1, 1, PE, 1]

            Returns:
                - torch.Tensor: Absolute-valued reconstructed image. [Batch, RO, PE]
                                Hint: To take the absolute value of complex numbers, you can use complex_abs function.
    """

    def __init__(self, num_cascades: int = 12, chans: int = 8, pools: int = 2, **kwargs):
        super().__init__()

        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        sens_reduce = kwargs.get("sens_reduce", Lab10_op.sens_reduce)
        varnet_block = kwargs.get("varnet_block", VarNetBlock)
        self.sens_reduce = sens_reduce

        # Your code here ...

    def forward(self, masked_kspace: torch.Tensor, sens_maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Your code here ...
        return None


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab10 import *

    op = Lab10_op()
