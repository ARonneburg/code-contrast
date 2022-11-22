from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Config:
    enc_name: str = "openai_reversible50000"
    T: int = 1024
    E: int = 384
    L: int = 4
    dtype_weights: str = "torch.float16"
    dtype_acts: str = "torch.float16"

    alt_sa_klass: Dict[str, Any] = field(default_factory=lambda: dict(type=''))
    alt_rel_klass: Dict[str, Any] = field(default_factory=lambda: dict(type=''))
    attn_seq: List[str] = field(default_factory=lambda: ['b', 'a'])
    attn_sparse_layout_seq: Optional[List] = None
    attn_heads: int = 16
    attn_ra_nbasis: int = 16
    attn_a_reach: int = 1024
    attn_b_reach: int = 160
    backcheck_sa: str = "none"  # FIXME not used

    alt_pw_klass: Dict[str, Any] = field(default_factory=lambda: dict(type=''))
    mlp_mult = 4

    rescale_embeddings: bool = False   # This means x_bte *= self.E ** 0.5 in embedding, not unembedding. Used in infill/fb.
    use_res_scale: bool = False
    unembedding_shared: bool = False
    posemb: bool = False
    backcheck_pw: str = "none"  # FIXME only 'inside' used
    _mup: bool = False
    mup_optimal_lr: Optional[float] = None
    mup_shapes_file: Optional[str] = None

    @property
    def n_vocab(self) -> int:
        return self.encoding.n_vocab

    @property
    def encoding(self):
        from code_contrast import Encoding
        return Encoding(self.enc_name)

    @property
    def mup(self) -> bool:
        return self._mup

    @mup.setter
    def mup(self, val: bool):
        from lean_former.mup import get_mup_information
        if val:
            maybe_info = get_mup_information(self)
            if not maybe_info:
                raise RuntimeError(f'No matching MUP layout has found for the following model hps')
            self._mup = True
            mup_shapes_file, self.mup_optimal_lr = maybe_info
            self.mup_shapes_file = os.path.relpath(mup_shapes_file, os.path.join(os.path.dirname(__file__), ".."))
        else:
            self._mup = False
            self.mup_shapes_file = None
            self.mup_optimal_lr = None

    def set_mup_manually(self, mup_shapes_file: str, mup_optimal_lr: float):
        assert os.path.exists(self.mup_shapes_file)
        self._mup = True
        self.mup_shapes_file = mup_shapes_file
        self.mup_optimal_lr = mup_optimal_lr