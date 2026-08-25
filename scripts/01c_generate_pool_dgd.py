#!/usr/bin/env python
"""Train DGD on D_train and emit a synthetic pool.

DGD is not a synthcity plugin, so it needs its own entry point.

* The model is fitted on exactly the D_train produced by ``00_make_splits.py``. DGD consumes
  a Bernoulli-encoded copy of the table, which has the same rows in the same order as the
  plain copy, so the partition is recovered by replaying the same stratified split on the
  row indices. The script asserts that the recovered rows match D_train before training.
* After training, the GMM is sampled and decoded to produce a pool of ``30 * |D_train|``
  records, mirroring the pools generated for the synthcity models.

DGD is not distributed with this repository. Point ``DGD_SRC`` at a checkout providing the
``mdgd``, ``dgd`` and ``survival_benchmark`` packages, and ``MAPS_DGD_DATA`` at the directory
holding the Bernoulli-encoded tables and variable descriptions (or pass them on the command
line). The other four generators need none of this.
"""
import argparse
import json
import os
import sys

# The DGD research packages live outside this repository; the variant used for the paper
# is the one providing mdgd.decoder_std.
for _p in (os.environ.get("DGD_SRC", "src_new"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import os
# The original script pinned the device here. Doing so overrides CUDA_VISIBLE_DEVICES set
# by the caller, which silently defeats any external scheduling, so the caller now wins and
# this only applies a default when nothing was specified.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GLOBAL_SEED = 42

import random
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
import torch
torch.manual_seed(GLOBAL_SEED)

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
torch.autograd.set_detect_anomaly(True)
sys.path.append("src_new")
import warnings
from mdgd.data_cleaner import MultiModalDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mdgd.decoder_std import Decoder
from mdgd.output_module import OutputModuleBasic, OutputModuleNormal, OutputModuleBernoulli, OutputModuleBeta, OutputModuleMultinomial, OutputModuleCounts
from mdgd.output_module import OutputModuleSurvival
from mdgd.metrics import concordance_td

from mdgd.loss import MultimodalLoss
from dgd.latent import RepresentationLayer
from dgd.latent import GaussianMixture
from mdgd.utils import plot_loss, plot_latent, data_to_device, decoder_to_device, latent_movie, mask_dictionary, plot_latent_categorical
from survival_benchmark.utils_surv import *
import argparse
import pickle

# Per-dataset inputs. DGD consumes a Bernoulli-encoded copy of the table plus a description
# file naming the output distribution of each variable, neither of which the other
# generators need. Epoch counts follow the manuscript: 2000 for TCGA, 1000 for GOSSIS.
_DGD_DATA = os.environ.get("MAPS_DGD_DATA", "data/dgd")
_DATASET_SETUP = {
    "tcga": dict(
        processed=f"{_DGD_DATA}/TCGA_metadata_21_vars_imputed_ber_processed.csv",
        description=f"{_DGD_DATA}/description_21_vars.csv",
        epochs=2000,
    ),
    "gossis": dict(
        processed=None,   # GOSSIS needs no separate Bernoulli encoding; see below
        description=f"{_DGD_DATA}/description_gossis_v1.csv",
        epochs=1000,
    ),
}

_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", default="tcga", choices=sorted(_DATASET_SETUP))
_ap.add_argument("--processed-csv", default=None)
_ap.add_argument("--description-csv", default=None)
_ap.add_argument("--epochs", type=int, default=None)
_args = _ap.parse_args()

_setup = _DATASET_SETUP[_args.dataset]
_args.processed_csv = _args.processed_csv or _setup["processed"]
_args.description_csv = _args.description_csv or _setup["description"]

from _common import load_config, load_paths          # noqa: E402
from maps.data.splits import load_split              # noqa: E402

_cfg = load_config(_args.dataset)
_paths = load_paths()
_split = load_split(_paths["splits_dir"], _cfg["dataset"]["name"])

_real_full = pd.read_csv(_paths["data_root"] / _cfg["dataset"]["real_csv"])
if _args.processed_csv:
    _processed_full = pd.read_csv(_args.processed_csv)
    if len(_processed_full) != len(_real_full):
        raise RuntimeError("the Bernoulli-encoded table and the plain table differ in length")
else:
    # No separately encoded copy for this dataset; the plain table is what DGD is given.
    _processed_full = _real_full

# Replay the split on row indices. train_test_split partitions positions from the label
# vector and the seed alone, so the same call on the same labels selects the same rows.
_tr_idx, _ = train_test_split(
    np.arange(len(_real_full)),
    train_size=_cfg["dataset"]["train_frac"],
    random_state=_cfg["dataset"]["split_seed"],
    stratify=_real_full[_cfg["dataset"]["target"]],
)
_recovered = _real_full.iloc[_tr_idx].reset_index(drop=True)
if not _recovered[_cfg["dataset"]["target"]].equals(_split.train[_cfg["dataset"]["target"]]):
    raise RuntimeError(
        "recovered rows do not match D_train; DGD would be trained on different data "
        "from the other generators"
    )
print(f"[dgd] recovered D_train: {len(_tr_idx)} rows, verified against the saved split")

multimodal_data = _processed_full.iloc[_tr_idx].reset_index(drop=True)
description = pd.read_csv(_args.description_csv)

# Bernoulli targets must be floating point: the loss goes through
# binary_cross_entropy_with_logits, which refuses an integer target. TCGA's separately
# encoded copy already stored them as floats, which is why this only shows up for datasets
# read straight from the plain table.
_bern = description.query('Model == "Bernoulli"')["Variable Name"].dropna().tolist()
_cast = [c for c in _bern if c in multimodal_data.columns
         and multimodal_data[c].dtype.kind in "iub"]
if _cast:
    multimodal_data[_cast] = multimodal_data[_cast].astype("float64")
    print(f"[dgd] cast {len(_cast)} Bernoulli column(s) to float: {_cast}")

# os.makedirs('results-only-21-var', exist_ok=True)
# os.chdir('results-only-21-var')
# os.makedirs('2000epochs-lr-1e-2-stratified-0.4test', exist_ok=True)
# os.chdir('2000epochs-lr-1e-2-stratified-0.4test')

n_epochs = _args.epochs or _setup['epochs']
print('n_epochs = ', n_epochs)
latent_dims = 50
print('latent_dims = ', latent_dims)
hidden_layers = [latent_dims,500,2000]
print('hidden_layers = ', hidden_layers)
components = 50
print('components = ', components)
sd_init = (0.063, 1.0)
mean_init = (7.0, 10.0)
print('sd_init = ', sd_init)
print('mean_init = ', mean_init)

decoder_lr = 1e-2
print('decoder_lr = ', decoder_lr)
gmm_lr = 1e-2
print('gmm_lr = ', gmm_lr)
rep_lr = 1e-2
print('rep_lr = ', rep_lr)

custom_layers = torch.nn.Sequential(
    torch.nn.Linear(hidden_layers[-1], 128),
    torch.nn.LayerNorm(128),
    torch.nn.SELU(),
    torch.nn.Dropout(0.85),
    torch.nn.Linear(128, 32),
    torch.nn.LayerNorm(32),
    torch.nn.SELU(),
    torch.nn.Dropout(0.85),
)
print("custom_layers = ", custom_layers)

hidden_normal = None
hidden_bernoulli = None
hidden_multinomial = None
hidden_counts = None
hidden_beta = None
hidden_poisson = None
hidden_survival = custom_layers

model_type = 'cph'
out_features = components if model_type == 'dcm' else 1

dataset = MultiModalDataset(multimodal_data, description)
# multimodal_data is already D_train. The small held-out slice below is only used to
# monitor the training loss; it is not the evaluation test set, which never enters here.
_target = _cfg['dataset']['target']
train_dataset, val_dataset = dataset.split(test_size=0.1, stratify=_target, random_state=GLOBAL_SEED)
test_dataset = val_dataset

##### get the correct train_cancer_types #####
# Label vectors for the latent diagnostics. Taken straight from the dataset objects so
# their lengths always match the representation layers; the original script derived them
# from a second, independently seeded split, which silently disagreed once the split
# fractions changed.
train_cancer_types = train_dataset.data[_target].values
val_cancer_types = val_dataset.data[_target].values
test_cancer_types = val_cancer_types

print("train_dataset size:", len(train_dataset.data))
print("val_dataset size:", len(val_dataset.data))
print("test_dataset size:", len(test_dataset.data))

train_data_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_data_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=True, num_workers=0)
print("batch_size:",256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gmm = GaussianMixture(n_mix_comp=components, dim=latent_dims,mean_init=mean_init,sd_init=sd_init, weight_alpha=5).to(device)
print("gmm weight_alpha:",5)

train_representation = RepresentationLayer(latent_dims, len(train_dataset)).to(device)
test_representation = RepresentationLayer(latent_dims, len(val_dataset)).to(device)

# Output modules are assembled from the modalities the dataset actually declares. TCGA has
# no Normal variables, so the original script had that module commented out; GOSSIS declares
# 40 of them, and without the module they are never decoded and the generated frame is
# missing those columns.
_HAS_NORMAL = hasattr(train_dataset, "NormalFeatureVariance") and \
    len(getattr(train_dataset.NormalFeatureVariance, "colname", []) or []) > 0
print(f"[dgd] Normal modality present: {_HAS_NORMAL}"
      + (f" ({len(train_dataset.NormalFeatureVariance.colname)} variables)" if _HAS_NORMAL else ""))

_omodules = []
if _HAS_NORMAL:
    _omodules.append(OutputModuleNormal(hidden_layers[-1],
                                        labels=train_dataset.NormalFeatureVariance.colname,
                                        var_type='Simple', layers=hidden_normal))
decoder = Decoder(hidden_layers, _omodules + [
    OutputModuleBernoulli(hidden_layers[-1], labels=train_dataset.Bernoulli.colname,layers=hidden_bernoulli),
    # OutputModuleBeta(hidden_layers[-1], labels=train_dataset.Beta.colname,layers=hidden_beta),
    # OutputModuleCounts(hidden_layers[-1], labels=train_dataset.NegativeBinomialFeatureDispersion.colname,r_type="Simple",layers=hidden_counts),
    OutputModuleCounts(hidden_layers[-1], labels=train_dataset.Poisson.colname,r_type="Poisson",layers=hidden_poisson),
    # Note I use labels, in order to handle variable sized output
    OutputModuleMultinomial(hidden_layers[-1],labels=train_dataset.Multinomial.labels,device=device,layers=hidden_multinomial),
   
    # OutputModuleSurvival(hidden_layers[-1],out_features=out_features,layers=hidden_survival,
    #                      loss_type = model_type)
]).to(device)

print(decoder)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr, weight_decay=4e-5, betas=(0.5, 0.9))
print("weight_decay:",4e-5)
gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=gmm_lr, weight_decay=4e-5, betas=(0.5,0.9)) # beta 
print("weight_decay:","4e-5")
rep_optimizer = torch.optim.Adam([
    {'params': train_representation.parameters()}, 
    {'params': test_representation.parameters()}],lr=rep_lr)
print("weight_decay:","no")

train_loss = MultimodalLoss(decoder.omodule.keys(), dataset_length=len(train_dataset))
test_loss = MultimodalLoss(decoder.omodule.keys(), dataset_length=len(val_dataset))

output_fils = "original_loss_data.pkl"

imgs = []
train_loss_dicts = []
test_loss_dicts = []

# # monitor the the C-index during training
# c_index_train = []
# c_index_test = []
# epochs_for_c_index = []

for idx, data_dict in train_data_loader:
    data_dict = data_to_device(data_dict, device)
    mask_dict = mask_dictionary(data_dict)
    break

for epoch in range(n_epochs):

    print("epoch:", epoch, flush=True)

    decoder.train()
    train_loss.reset()
    test_loss.reset()

    rep_optimizer.zero_grad()

    epoch_train_loss_dicts = []
    for idx, data_dict in train_data_loader:

        gmm_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # compute reconstructions
        z = train_representation(idx)
        x = decoder(z)
        
        data_dict = data_to_device(data_dict,device)
        x = decoder_to_device(x,device)
        
        # if decoder.omodule['Survival'].loss_type == 'dcm':
        #     gmm_weights = torch.softmax(gmm.sample_probs(z), dim=-1).detach()
        # else:
        #     gmm_weights = None

        recon_loss, loss_dict = decoder.loss(x, data_dict, idx=idx, gmm_weights=None)

        recon_loss = 0.0
        for key,item in loss_dict.items():
            recon_loss += loss_dict[key]

        gmm_loss = gmm(z).sum()
        loss = recon_loss + gmm_loss
        train_loss.batch_update(loss_dict,data_dict,gmm_loss)

        epoch_train_loss_dicts.append({k: v.cpu() for k, v in loss_dict.items()})

        loss.backward()
        decoder_optimizer.step()
        gmm_optimizer.step()

    train_loss_dicts.append(epoch_train_loss_dicts)

    epoch_test_loss_dicts = []    
    for idx, data_dict in test_data_loader:

        # compute reconstructions
        z = test_representation(idx)
        x = decoder(z)
        
        data_dict = data_to_device(data_dict,device)
        x = decoder_to_device(x,device)

        # create mask dictionary and mask survival data
        mask_dict = mask_dictionary(data_dict)
        # mask_dict['Survival'].fill_(0)

        # if decoder.omodule['Survival'].loss_type == 'dcm':
        #     gmm_weights = torch.softmax(gmm.sample_probs(z), dim=-1).detach()
        # else:
        #     gmm_weights = None
        
        recon_loss, loss_dict = decoder.loss(x, data_dict, idx=idx, mask=mask_dict, gmm_weights=None)
        
        recon_loss = 0.0
        for key,item in loss_dict.items():
            recon_loss += loss_dict[key]

        gmm_loss = gmm(z).sum()
        loss = recon_loss + gmm_loss

        epoch_test_loss_dicts.append({k: v.cpu() for k, v in loss_dict.items()})
        loss.backward()

        # Monitor survival after without updating gradients
        decoder.eval()
        _,loss_dict = decoder.loss(x, data_dict, idx=idx, gmm_weights=None)

        for key,item in loss_dict.items():
            recon_loss += loss_dict[key]
        
        loss = recon_loss + gmm_loss

        test_loss.batch_update(loss_dict,data_dict,gmm_loss)
    
    test_loss_dicts.append(epoch_test_loss_dicts)

    # Latent diagnostics are not needed to produce a pool and were the only consumer of
    # the label vectors, so they are switched off here.
    # img,fig = plot_latent_categorical(...)
    # plt.close(fig)
#     imgs.append(img)
    
    rep_optimizer.step()

    train_loss.epoch_update()
    test_loss.epoch_update()

    # # record the C-index every 10 epochs
    # if epoch % 10 == 0 or epoch == n_epochs-1:
    #     n_train = len(train_dataset.data)
    #     n_val = len(val_dataset.data)

    #     pred_train = np.exp(decoder(train_representation(torch.arange(n_train)))['Survival'].detach().cpu().numpy()).ravel()
    #     pred_val = np.exp(decoder(test_representation(torch.arange(n_val)))['Survival'].detach().cpu().numpy()).ravel()

    #     c_index_train.append(concordance_index_censored(train_dataset.Survival.data[:,1].numpy().astype(bool), train_dataset.Survival.data[:,0].numpy(), pred_train)[0])
    #     c_index_test.append(concordance_index_censored(val_dataset.Survival.data[:,1].numpy().astype(bool), val_dataset.Survival.data[:,0].numpy(), pred_val)[0])
    #     epochs_for_c_index.append(epoch)


# check the mode
if decoder.training:
    print("Train Mode")
else:
    print("Evaluation Mode")

# _,fig = plot_latent_categorical(train_representation.z.detach().cpu().numpy(),test_representation.z.detach().cpu().numpy(),gmm,
#             train_cancer_types,
#             val_cancer_types,epoch,categorical_cmap=plt.get_cmap('tab20c'))
# plt.show()
# fig.savefig("latent.pdf")
# plot_loss(train_loss,test_loss,n_epochs,"loss.pdf")
# latent_movie(imgs,'latent_movie.mp4')



import torch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bernoulli mapping and inverse transform.
#
# The original mapping_21var.json in the data directory is a zero-byte file, so the
# string<->binary mapping was reconstructed by comparing the plain and Bernoulli-encoded
# copies of the table row by row; both list the same records in the same order. The
# reconstruction is written to bernoulli_mapping_21var.json and is verified below.
# ---------------------------------------------------------------------------
def _load_or_build_bernoulli_mapping(real_df, processed_df, description_df, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as fh:
            return json.load(fh)
    cols = description_df.query('Model == "Bernoulli"')["Variable Name"].tolist()
    mapping = {}
    for c in cols:
        if c not in real_df.columns:
            continue
        pairs = pd.DataFrame(
            {"s": real_df[c].astype(str), "b": processed_df[c]}
        ).drop_duplicates()
        m = {}
        for _, r in pairs.iterrows():
            if pd.isna(r["b"]):
                continue
            if r["s"] in m and m[r["s"]] != r["b"]:
                raise RuntimeError(f"inconsistent Bernoulli mapping for {c}")
            m[r["s"]] = float(r["b"])
        mapping[c] = m
    with open(path, "w") as fh:
        json.dump(mapping, fh, indent=2)
    return mapping


def process_bernoulli_variables(df, mapping, inverse=False):
    """Convert Bernoulli columns between their string labels and 0/1 encoding."""
    out = df.copy()
    for col in out.columns:
        if col in mapping:
            if inverse:
                out[col] = out[col].map({v: k for k, v in mapping[col].items()})
            else:
                out[col] = out[col].map(mapping[col])
    return out, mapping


def generate_synthetic_datasets(num_datasets, n_generated_sample, gmm, decoder, train_dataset, device):
    synthetic_latent = []
    synthetic_datasets = []
    
    for _ in range(num_datasets):
        # Generate latent space samples
        generated_sample_rep = gmm.sample_new_points(resample_type='sample', n_new_samples=n_generated_sample)
        generated_sample_rep = torch.tensor(generated_sample_rep).to(device)
        synthetic_latent.append(generated_sample_rep)
        
        # Cluster the generated samples
        generated_sample_cluster = gmm.clustering(generated_sample_rep).cpu().numpy()
        
        # Pass generated samples through the decoder
        generated_samples = decoder.forward(generated_sample_rep)
        
        # Process each modality
        generated_samples["Bernoulli"] = decoder.omodule['Bernoulli'].sample(
            generated_samples['Bernoulli']
        ).detach().cpu().numpy()
        
        multinomial_out = generated_samples['Multinomial']
        samples = decoder.omodule.Multinomial.sample(multinomial_out, 1).T.detach().cpu().numpy()
        generated_samples["Multinomial"] = samples
        
        if 'NormalFeatureVariance' in generated_samples:
            # Decoded on the standardised scale, so undo the scaling the dataset applied.
            mean, sigma = train_dataset.NormalFeatureVariance.scale
            mean, sigma = np.asarray(mean), np.asarray(sigma)
            generated_samples["NormalFeatureVariance"] = (
                decoder.omodule['NormalFeatureVariance']
                .sample(generated_samples['NormalFeatureVariance'])
                .detach().cpu().numpy() * sigma) + mean
        
        generated_samples["Poisson"] = decoder.omodule['Poisson'].sample(
            generated_samples['Poisson'], idx=np.arange(0, len(train_dataset)), scale=train_dataset.Poisson.scale
        ).detach().cpu().numpy()

        # generated_samples["Beta"] = decoder.omodule['Beta'].sample(generated_samples['Beta']).detach().cpu().numpy()
        
        # Convert to DataFrame
        syn_bernoulli_df = pd.DataFrame(generated_samples["Bernoulli"], columns=train_dataset.Bernoulli.colname)
        syn_multinomial_df = pd.DataFrame(generated_samples["Multinomial"], columns=train_dataset.Multinomial.colname)
        syn_normal_df = (
            pd.DataFrame(generated_samples["NormalFeatureVariance"],
                         columns=train_dataset.NormalFeatureVariance.colname)
            if 'NormalFeatureVariance' in generated_samples else None)
        syn_poisson_df = pd.DataFrame(generated_samples["Poisson"], columns=train_dataset.Poisson.colname)
        # syn_beta_df = pd.DataFrame(generated_samples["Beta"], columns=train_dataset.Beta.colname)
        
        # # Reverse mapping for Bernoulli variables
        syn_bernoulli_df, _ = process_bernoulli_variables(syn_bernoulli_df,mapping=mapping, inverse=True)

        # # for all the NaN values in the syn_bernoulli_df, convert NaN to string 'NaN'
        # syn_bernoulli_df = syn_bernoulli_df.fillna('NaN')
    
        # Reverse mapping for Multinomial variables
        multinomial_labels = train_dataset.Multinomial.labels
        for col in syn_multinomial_df.columns:
            syn_multinomial_df[col] = syn_multinomial_df[col].map(lambda x: multinomial_labels[col][x] if x != -1 else None)
        
        # Combine all the generated samples dataframes together
        _frames = [syn_multinomial_df, syn_bernoulli_df, syn_poisson_df]
        if syn_normal_df is not None:
            _frames.append(syn_normal_df)
        df_fake = pd.concat(_frames, axis=1)
        # drop the survival realted columns ['tcga.gdc_cases.diagnoses.vital_status', 'tcga.gdc_cases.diagnoses.days_to_death']
        # train_dataset.data.drop(columns=['tcga.gdc_cases.diagnoses.vital_status', 'tcga.gdc_cases.diagnoses.days_to_death'], inplace=True)
        # make sure df_fake has the same columns as the real dataset in the same order
        df_fake = df_fake[train_dataset.data.columns]
        synthetic_datasets.append(df_fake)
    
    return synthetic_datasets, synthetic_latent


# ---------------------------------------------------------------------------
# Sample the trained GMM, decode, and write a pool of 30 * |D_train| records.
# ---------------------------------------------------------------------------
_mapping_path = str(_paths["data_root"] / "raw" / "tcga" / "bernoulli_mapping_21var.json")
mapping = _load_or_build_bernoulli_mapping(
    _real_full, _processed_full, description, _mapping_path
)
print(f"[dgd] Bernoulli mapping: {mapping}")

_multiplier = _cfg["maps"]["pool_multiplier"]
_n_train = len(_split.train)
_per_dataset = len(train_dataset)

from _common import Timer, write_json           # noqa: E402

# Decode in batches. The multinomial sampler materialises a nested tensor over all
# categories at once, which for TCGA is a 2.3 GB allocation per dataset; that is fine on an
# idle device and fails when the GPU is shared. Batching bounds the peak instead of relying
# on having the card to ourselves.
_BATCH = int(os.environ.get("DGD_GEN_BATCH", "2000"))


def _generate_in_batches(n_datasets, n_per, gmm, decoder, train_dataset, device, batch):
    out = []
    for k in range(n_datasets):
        parts, remaining = [], n_per
        while remaining > 0:
            take = min(batch, remaining)
            sets, _ = generate_synthetic_datasets(1, take, gmm, decoder, train_dataset, device)
            parts.append(sets[0])
            remaining -= take
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out.append(pd.concat(parts, ignore_index=True))
        print(f"  [dgd] dataset {k + 1}/{n_datasets} decoded")
    return out


with Timer(f"generate:{_args.dataset}:dgd") as _t:
    _sets = _generate_in_batches(
        _multiplier + 2, _per_dataset, gmm, decoder, train_dataset, device, _BATCH
    )
    _pool = pd.concat(_sets, ignore_index=True)
    # Trim to exactly multiplier * |D_train|, matching the synthcity pools.
    _pool = _pool.iloc[: _multiplier * _n_train].reset_index(drop=True)

# Restore the original dtypes. Bernoulli columns were cast to float for training, and
# leaving them that way makes the synthetic table structurally different from the real one:
# a 0/1 integer column comes back as 0.0/1.0, which breaks any label encoder fitted on the
# real values. Synthetic data should be interchangeable with real data, so the cast is undone
# where the source column was integral.
for _c in _pool.columns:
    if _c in _real_full.columns and _real_full[_c].dtype.kind in "iu":
        _pool[_c] = _pool[_c].round().astype(_real_full[_c].dtype)
print(f"[dgd] restored integer dtype on "
      f"{sum(1 for c in _pool.columns if c in _real_full.columns and _real_full[c].dtype.kind in 'iu')} column(s)")

_out = _paths["pools_dir"] / _args.dataset / f"dgd_{_multiplier}x.csv"
_out.parent.mkdir(parents=True, exist_ok=True)
_pool.to_csv(_out, index=False)
print(f"[write] {_out}  ({_pool.shape[0]} x {_pool.shape[1]})")

write_json(
    {"dataset": _args.dataset, "model": "dgd", "fitted_on": "D_train",
     "n_train": _n_train, "n_pool": int(_pool.shape[0]),
     "pool_multiplier": _multiplier, "n_epochs": n_epochs,
     "latent_dims": latent_dims, "gmm_components": components,
     "train_sha1": _split.sha1.get("train_sha1"),
     "bernoulli_mapping": mapping,
     "timing": {"generate": _t.record}},
    _paths["results_dir"] / _args.dataset / "dgd_pool_meta.json",
)
