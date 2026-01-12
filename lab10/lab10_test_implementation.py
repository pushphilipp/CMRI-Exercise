#!/usr/bin/env python
# coding: utf-8

# Connected to cmri (Python 3.9.0)

# In[1]:


# This import is necessary to run the code cell-by-cell
from lab10 import *

op = Lab10_op()

# --- Save all figures produced by this script into a dedicated folder
from pathlib import Path

_OUT_DIR = Path(__file__).parent / "extracted_plots_own_implementatin"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_plot_counter = 0


def _next_plot_filename() -> str:
    """Deterministic plot name (matches the solution extractor style)."""
    global _plot_counter
    _plot_counter += 1
    return f"plot_{_plot_counter}.png"


# In[8]:


dataset = FastMRIDataset(op.data_root / "train", prototype=False)
dataset_prot = FastMRIDataset(op.data_root / "train", prototype=True)

print(len(dataset), len(dataset_prot))


# In[7]:


data_path1 = list((op.data_root / "train").iterdir())[0]
print(data_path1)

with h5py.File(data_path1) as hf:
    print(hf.keys())


# In[2]:


dataset = FastMRIDataset(op.data_root / "train", prototype=True)[0]
print(f"Fields in the dataset: {dataset._fields}")
print(f"Shape of masked k-space: {dataset.masked_kspace.shape}")
print(f"Shape of mask: {dataset.mask.shape}")
print(f"Shape of sensitivity maps: {dataset.sens_maps.shape}")
print(f"Shape of target: {dataset.target.shape}")


# In[3]:


utils.imshow(
    [dataset.masked_kspace[0]],
    norm=0.3,
    titles=["Undersampled kspace"],
    root=_OUT_DIR,
    filename=_next_plot_filename(),
)
utils.imshow(
    [dataset.target],
    titles=["Ground truth"],
    root=_OUT_DIR,
    filename=_next_plot_filename(),
)


# In[ ]:


utils.pprint("- Load data and create dataloaders")
train_dataloader = DataLoader(FastMRIDataset(op.data_root / "train"), batch_size=4, shuffle=True)
validation_dataloader = DataLoader(FastMRIDataset(op.data_root / "val"), batch_size=4, shuffle=False)
test_dataloader = DataLoader(FastMRIDataset(op.data_root / "test"), batch_size=4, shuffle=False)

utils.pprint(f"- Train data: {len(train_dataloader.dataset)}", level=1)
utils.pprint(f"- Validation data: {len(validation_dataloader.dataset)}", level=1)
utils.pprint(f"- Test data: {len(test_dataloader.dataset)}", level=1)


# ## To define cascades in Variational network
# cascades =[VarnetBlock(Unet(pools, chans)) for i in num_cascades]

# In[5]:


model_params = {"UNet": {"chans": 12, "pools": 4}, "VarNet": {"num_cascades": 5, "chans": 16, "pools": 4}}

model_name = "UNet"
model = op.get_model(model_name=model_name, **model_params[model_name])
op.tester(
    model,
    test_dataloader,
    verbose=True,
    model_name=model_name,
    checkpoint_path=PRET_WEIGHTS_UNET,
)
model_name = "VarNet"
model = op.get_model(model_name=model_name, **model_params[model_name])
op.tester(
    model,
    test_dataloader,
    verbose=True,
    model_name=model_name,
    checkpoint_path=PRET_WEIGHTS_VARNET,
)

files = np.unique([i.stem.split("_")[1] for i in (op.save_dir / "Pretrained/Results").iterdir()])
for file in files:
    unet_recon = np.load(op.save_dir / "Pretrained/Results" / f"UNet_{file}.npy")
    varnet_recon = np.load(op.save_dir / "Pretrained/Results" / f"VarNet_{file}.npy")
    target = np.load(op.save_dir / "Pretrained/Results" / f"target_{file}.npy")
    utils.imshow(
        [unet_recon, varnet_recon],
        titles=["UNet", "VarNet"],
        gt=target,
        suptitle=file,
        root=_OUT_DIR,
        filename=_next_plot_filename(),
    )


# In[10]:


model_params = {"UNet": {"chans": 12, "pools": 4}, "VarNet": {"num_cascades": 5, "chans": 16, "pools": 4}}

learning_rate = 1e-3
epochs = 10

## U-Net
utils.pprint("- selcting a model")
seed_everything()
model_name = "UNet"
model = op.get_model(model_name=model_name, **model_params[model_name])
print(model)


# In[6]:


learning_rate = 1e-3
epochs = 10

## U-Net
utils.pprint("- selcting a model")
seed_everything()
model_name = "UNet"
model = op.get_model(model_name=model_name, **model_params[model_name])
criterion = op.get_loss()
optimizer = op.get_optimizer(model, learning_rate)

utils.pprint("- Start training")
unet_train_loss, unet_validation_loss = op.trainer(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=epochs,
    model_name=model_name,
    checkpoint_path=PRET_WEIGHTS_UNET,
)

utils.pprint("- Start testing")
op.tester(model, test_dataloader, verbose=True, model_name=model_name)

## VarNet
utils.pprint("- selcting a model")
seed_everything()
model_name = "VarNet"
model = op.get_model(model_name=model_name, **model_params[model_name])
criterion = op.get_loss()
optimizer = op.get_optimizer(model, learning_rate)

utils.pprint("- Start training")
vn_train_loss, vn_validation_loss = op.trainer(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=epochs,
    model_name=model_name,
    checkpoint_path=PRET_WEIGHTS_VARNET,
)

utils.pprint("- Start testing")
op.tester(model, test_dataloader, verbose=True, model_name=model_name)


# In[7]:


utils.plot(
    [unet_train_loss, unet_validation_loss],
    labels=["train loss", "validation loss"],
    title=f"UNet loss",
    xlabel="Epoch",
    ylabel="Loss",
    root=_OUT_DIR,
    filename=Path(_next_plot_filename()).stem,
)
utils.plot(
    [vn_train_loss, vn_validation_loss],
    labels=["train loss", "validation loss"],
    title=f"VarNet loss",
    xlabel="Epoch",
    ylabel="Loss",
    root=_OUT_DIR,
    filename=Path(_next_plot_filename()).stem,
)


# In[8]:


files = np.unique([i.stem.split("_")[1] for i in (op.save_dir / "Pretrained/Results").iterdir()])
for file in files:
    unet_recon_pret = np.load(op.save_dir / "Pretrained/Results" / f"UNet_{file}.npy")
    varnet_recon_pret = np.load(op.save_dir / "Pretrained/Results" / f"VarNet_{file}.npy")
    unet_recon_finetune = np.load(op.save_dir / "Pretrained/Results" / f"UNet_{file}.npy")
    varnet_recon_finetune = np.load(op.save_dir / "Pretrained/Results" / f"VarNet_{file}.npy")
    target = np.load(op.save_dir / "Pretrained/Results" / f"target_{file}.npy")
    utils.imshow(
        [unet_recon_pret, varnet_recon_pret, unet_recon_finetune, varnet_recon_finetune],
        titles=["UNet (Pretrained)", "VarNet (Pretrained)", "UNet (Fine tuned)", "VarNet (Fine tuned)"],
        gt=target,
        suptitle=file,
        num_rows=2,
        root=_OUT_DIR,
        filename=_next_plot_filename(),
    )
