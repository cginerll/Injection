"""
methanex.py

Methane column enhancement retrieval from imaging spectrometer data.

    eng = MethaneRetrieval(device="cuda")

    mf, R, meta = eng.retrieve(rad, MFConfig())
    mf, R, meta = eng.retrieve(rad, RMFConfig())
    mf, R, meta = eng.retrieve(rad, MAG1CConfig())
    mf, R, meta = eng.retrieve(rad, MAG1CFastConfig(tol=1e-5))

rad must be a torch.Tensor of shape (downtrack, crosstrack, bands).
retrieve() returns a RetrievalResult that supports both named access
and tuple unpacking (mf, albedo, metadata).

The CH4 absorption template for EMIT is loaded automatically from:
    https://data.source.coop/taco/methaneset/ch4_emit.safetensors
"""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from tqdm import tqdm


# Constants --------------------

SAFETENSORS_URL  = "https://data.source.coop/taco/methaneset/ch4_emit.safetensors"
SAFETENSORS_PATH = Path(__file__).resolve().parent / "ch4_emit.safetensors"
WAVELENGTH_RANGE = (2122, 2488)
EPSILON          = 1e-9
NODATA           = -9999


# Result container --------------------

@dataclass
class RetrievalResult:
    """Output of MethaneRetrieval.retrieve().

    Supports both named access and tuple unpacking:
        result = eng.retrieve(rad, config)
        result.mf, result.albedo, result.metadata

        mf, albedo, meta = eng.retrieve(rad, config)
        mf, albedo, _    = eng.retrieve(rad, config)
    """
    mf: torch.Tensor
    albedo: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.mf, self.albedo, self.metadata))

    def __getitem__(self, idx):
        return (self.mf, self.albedo, self.metadata)[idx]

    def __len__(self):
        return 3


# Config models --------------------

class BaseRetrievalConfig(BaseModel):
    """Parameters shared by all retrieval methods."""

    column_step: Annotated[int, Field(ge=1, description=(
        "Number of crosstrack columns processed together per engine call. "
        "column_step=1 is physically correct for push-broom sensors. "
        "Ignored when batch_size is set (forced to 1)."
    ))] = 1

    batch_size: Annotated[int | None, Field(ge=1, description=(
        "Number of independent columns processed in parallel. "
        "When set, column_step is forced to 1. "
        "NODATA pixels are filled with per-column mean and re-masked on output. "
        "None = legacy path with explicit NODATA filtering."
    ))] = None

    alpha: Annotated[float, Field(ge=0.0, le=1.0, description=(
        "Covariance diagonal regularisation weight. "
        "Small values (~1e-4) stabilise the Cholesky decomposition."
    ))] = 1e-4

    scaling: Annotated[float, Field(gt=0.0, description=(
        "Multiplicative factor applied to raw matched filter output. "
        "Default 1e5 maps EMIT radiance-scale retrievals to ~ppm*m."
    ))] = 1e5


class MFConfig(BaseRetrievalConfig):
    """Basic matched filter (no albedo correction)."""
    pass


class RMFConfig(BaseRetrievalConfig):
    """Albedo-corrected single-pass matched filter."""
    pass


class MAG1CConfig(BaseRetrievalConfig):
    """Iterative sparse matched filter (reference implementation)."""

    num_iter: Annotated[int, Field(ge=1, description=(
        "Number of reweighted-L1 iterations."
    ))] = 30

    covariance_update_scaling: Annotated[float, Field(ge=0.0, le=1.0, description=(
        "Fraction of estimated CH4 signal subtracted at each step. "
        "1.0 = full removal, 0.0 = no update."
    ))] = 1.0


class MAG1CFastConfig(MAG1CConfig):
    """Rank-2 accelerated iterative filter."""

    tol: Annotated[float | None, Field(ge=0.0, description=(
        "Early stopping on max per-column relative mf change. "
        "None = disabled, runs exactly num_iter steps."
    ))] = None

    outer_mode: Annotated[Literal["auto", "bmm", "einsum"], Field(description=(
        "How to compute the rank-2 correction. "
        "'auto' picks bmm when N < 2000 or device is CPU."
    ))] = "auto"


RetrievalConfig = MFConfig | RMFConfig | MAG1CConfig | MAG1CFastConfig

_DISPATCH: dict[type, str] = {
    MFConfig:        "mf",
    RMFConfig:       "rmf",
    MAG1CConfig:     "mag1c",
    MAG1CFastConfig: "mag1c_fast",
}


# Template helpers --------------------

def _download_template(url: str, dest: Path) -> None:
    print(f"Downloading CH4 template to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def _load_template(path, wavelength_range, device, dtype):
    if not path.exists():
        _download_template(SAFETENSORS_URL, path)
    tensors   = load_file(str(path), device=device)
    centers   = tensors["centers"]
    template  = tensors["template"]
    band_mask = (centers >= wavelength_range[0]) & (centers <= wavelength_range[1])
    return template[band_mask].to(dtype), band_mask.cpu()


# NODATA helpers --------------------

def _fill_nodata(rad):
    """Replace NODATA pixels with per-column mean. Fully vectorized.

    Returns:
        filled:      Clone with NODATA replaced by per-column mean.
        nodata_mask: Bool (D, C), True where any band was NODATA.
        empty_cols:  Bool (C,), True for entirely-NODATA columns.
    """
    filled      = rad.clone()
    nodata_mask = ((filled == NODATA) | torch.isnan(filled)).any(dim=-1)
    filled[nodata_mask] = 0.0

    counts     = (~nodata_mask).sum(dim=0)
    empty_cols = counts == 0

    col_sum  = filled.sum(dim=0, keepdim=True)
    col_mean = col_sum / counts.unsqueeze(0).clamp(min=1).unsqueeze(-1)
    filled[nodata_mask] = col_mean.expand_as(filled)[nodata_mask]

    return filled, nodata_mask, empty_cols


# Retrieval engine --------------------

class MethaneRetrieval:
    """Methane column enhancement retrieval engine."""

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        wavelength_range: tuple[float, float] = WAVELENGTH_RANGE,
        safetensor_path: Path = SAFETENSORS_PATH,
        alpha: float = 1e-4,
        num_iter: int = 30,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.device   = device
        self.dtype    = dtype
        self.alpha    = alpha
        self.num_iter = num_iter
        self.template, self.band_mask = _load_template(
            safetensor_path,
