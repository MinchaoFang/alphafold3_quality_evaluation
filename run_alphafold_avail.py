# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Iterable, Sequence
import csv
import dataclasses
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import Final, Protocol, Self, TypeVar, overload
from absl import app
from absl import flags
import sys
sys.path.insert(0, '/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation')

# 检查结果
print(sys.path)
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import base_model
from alphafold3.model.components import utils
from alphafold3.model.diffusion import model as diffusion_model
print(alphafold3.cpp.__file__)
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
#jax.config.update("jax_disable_jit", True)
import numpy as np
from typing import Dict, Any, Optional
import pickle

_HOME_DIR = pathlib.Path("root")
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_REF_PDB_PATH = flags.DEFINE_string(
    'ref_pdb_path',
    None,
    'Path to the input ref pdb file.',
)
_REF_TIME_STEPS = flags.DEFINE_integer(
    'ref_time_steps',
    0,
    'total 200 steps, input the steps you want',
)
_REF_PKL_DUMP_PATH =  flags.DEFINE_string(
    'ref_pkl_dump_path',
    None,
    'Path to dumped pkl path, for debug only',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    False,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
_DB_DIR = flags.DEFINE_string(
    'db_dir',
    _DEFAULT_DB_DIR.as_posix(),
    'Path to the directory containing the databases.',
)
_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/pdb_2022_09_28_mmcif_files.tar',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Compilation cache
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)

_BUCKETS: Final[tuple[int, ...]] = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)
# Atom mapping directly
DENSE_ATOM = {
    # Protein
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": (),
    # RNA
    "A": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "C": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "G": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "U": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    "UNK_RNA": (),
    # DNA
    "DA": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "DC": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "DG": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "DT": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
    "UNK_DNA": (),
}
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional

class StructureParser:
    def __init__(self):
        self.ATOM_RECORD_FORMAT = {
            'record_name': (0, 6),
            'atom_number': (6, 11),
            'atom_name': (12, 16),
            'alt_loc': (16, 17),
            'res_name': (17, 20),
            'chain_id': (21, 22),
            'res_number': (22, 26),
            'x': (30, 38),
            'y': (38, 46),
            'z': (46, 54),
            'occupancy': (54, 60),
            'temp_factor': (60, 66),
            'element': (76, 78)
        }
    
    def parse_pdb(self, file_path: str) -> Dict[str, List]:
        """解析PDB文件并提取原子信息"""
        coords=0
                        
        return coords
    
    def parse_cif(self, file_path: str) -> Dict[str, List]:
        """解析mmCIF文件并提取原子信息"""
        coords = 0
        
        return coords
    
           

    def process_file(self, file_path: str) -> Dict:
        """处理完整的文件并返回JAX可用的数据"""
        if file_path.endswith('.pdb'):
            coords = self.parse_pdb(file_path)
        elif file_path.endswith('.cif'):
            coords = self.parse_cif(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, "rb") as f:
                coords = pickle.load(f)
        else:
            raise ValueError("文件格式必须是.pdb or .cif or .pkl")
            
        return jnp.array(coords)



class StructureFeatureIntegrator:
    @staticmethod
    def process_ref_file(ref_pdb_path: str) -> Dict[str, Any]:
        """处理参考结构文件并提取特征"""
        parser = StructureParser()
        coords = parser.process_file(ref_pdb_path)
        # 提取和处理参考结构特征
        ref_features = {'ref_atom_positions': coords
        }
        return ref_features
    
    @staticmethod
    def integrate_reference_features(
        featurised_example: Dict[str, Any],
        ref_structure_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将参考结构特征集成到已特征化的示例中"""
        # 深拷贝以避免修改原始数据
        integrated_features = dict(featurised_example)
        
        # 添加参考结构特征
        for key, value in ref_structure_data.items():
            integrated_features[key] = value
            
        
        return integrated_features
    
    
    @staticmethod
    def process_and_integrate(
        featurised_example: Dict[str, Any],
        ref_pdb_path: str,
        device: Optional[jax.Device] = None
    ) -> Dict[str, Any]:
        """完整的处理和集成流程"""
        device = device or jax.devices()[0]
        # 处理参考结构
        ref_structure_data = StructureFeatureIntegrator.process_ref_file(ref_pdb_path)
        
        # 集成特征
        integrated_features = StructureFeatureIntegrator.integrate_reference_features(
            featurised_example,
            ref_structure_data
        )

        # 转换为设备数组
        device_features = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray,
                integrated_features
            ),
            device
        )
        return device_features
class ConfigurableModel(Protocol):
  """A model with a nested config class."""

  class Config(base_config.BaseConfig):
    ...

  def __call__(self, config: Config) -> Self:
    ...

  @classmethod
  def get_inference_result(
      cls: Self,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str = '',
  ) -> Iterable[base_model.InferenceResult]:
    ...


ModelT = TypeVar('ModelT', bound=ConfigurableModel)


def make_model_config(
    *,
    model_class: type[ModelT] = diffusion_model.Diffuser,
    flash_attention_implementation: attention.Implementation = 'triton',
    ref_time_steps= 0,
):
  config = model_class.Config()

  if hasattr(config, 'global_config'):
    config.global_config.flash_attention_implementation = (
        flash_attention_implementation
    )
  if ref_time_steps !=0:
    config.heads.diffusion.eval.ref_time_steps = ref_time_steps
    config.heads.diffusion.eval.num_samples = 1
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      model_class: ConfigurableModel,
      config: base_config.BaseConfig,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_class = model_class
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], base_model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""
    assert isinstance(self._model_config, self._model_class.Config)

    @hk.transform
    def forward_fn(batch):
      result = self._model_class(self._model_config)(batch)
      result['__identifier__'] = self.model_params['__meta__']['__identifier__']
      return result

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
    self, 
    featurised_example: features.BatchDict, 
    rng_key: jnp.ndarray, 
    ref_pdb_path: os.PathLike[str] | None,  # 为 ref_pdb_path 添加类型注释
    ref_pkl_dump_path: os.PathLike[str] | None,
) -> base_model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    if ref_pdb_path :
      print(f'Running ref_pdb guided prediction '+ref_pdb_path)
      if not os.path.exists(ref_pdb_path):
        raise FileNotFoundError(f"Ref PDB file not found: {ref_pdb_path}")
      integrator = StructureFeatureIntegrator()
      final_features = integrator.process_and_integrate(
    featurised_example,
    ref_pdb_path,
      )
      result = self._model(rng_key, final_features)
    else:
      print(f'No ref_pdb guided prediction')

      result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result['__identifier__'] = result['__identifier__'].tobytes()
    print(result["diffusion_samples"]["atom_positions"].shape)
    if ref_pkl_dump_path:
        print(f"dump pkl file in {ref_pkl_dump_path}")
        with open(ref_pkl_dump_path, 'wb') as f:
            pickle.dump(np.array(jax.device_get(result["diffusion_samples"]["atom_positions"])), f)
    return result

  def extract_structures(
      self,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str,
  ) -> list[base_model.InferenceResult]:
    """Generates structures from model outputs."""
    return list(
        self._model_class.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
  """

  seed: int
  inference_results: Sequence[base_model.InferenceResult]
  full_fold_input: folding_input.Input


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    ref_pdb_path: os.PathLike[str] | str,
    ref_pkl_dump_path: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data for seeds {fold_input.rng_seeds}...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
  )
  print(
      f'Featurising data for seeds {fold_input.rng_seeds} took '
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference for seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key,ref_pdb_path,ref_pkl_dump_path)
    print(
        f'Running model inference for seed {seed} took '
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting output structures (one per sample) for seed {seed}...')
    extract_structures = time.time()
    inference_results = model_runner.extract_structures(
        batch=example, result=result, target_name=fold_input.name
    )
    print(
        f'Extracting output structures (one per sample) for seed {seed} took '
        f' {time.time() - extract_structures:.2f} seconds.'
    )
    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
        )
    )
    print(
        'Running model inference and extracting output structures for seed'
        f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
    )
  print(
      'Running model inference and extracting output structures for seeds'
      f' {fold_input.rng_seeds} took '
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  with open(
      os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
  ) as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result, output_dir=sample_dir,name=f'seed-{seed}_sample-{sample_idx}'
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    ref_pdb_path: os.PathLike[str] | None,
    ref_pkl_dump_path: os.PathLike[str] | None,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'Processing fold input {fold_input.name}')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if model_runner is not None:
    # If we're running inference, check we can load the model parameters before
    # (possibly) launching the data pipeline.
    print('Checking we can load the model parameters...')
    _ = model_runner.model_params

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  print(f'Output directory: {output_dir}')
  print(f'Writing model input JSON to {output_dir}')
  write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        ref_pdb_path=ref_pdb_path,
        ref_pkl_dump_path=ref_pkl_dump_path,
        buckets=buckets)
    print(
        f'Writing outputs for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
    )
    output = all_inference_results

  print(f'Done processing fold input {fold_input.name}.')
  return output


def main(_):
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )

  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n'.join(notice))

  if _RUN_DATA_PIPELINE.value:
    replace_db_dir = lambda x: string.Template(x).substitute(
        DB_DIR=_DB_DIR.value
    )
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
        small_bfd_database_path=replace_db_dir(_SMALL_BFD_DATABASE_PATH.value),
        mgnify_database_path=replace_db_dir(_MGNIFY_DATABASE_PATH.value),
        uniprot_cluster_annot_database_path=replace_db_dir(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniref90_database_path=replace_db_dir(_UNIREF90_DATABASE_PATH.value),
        ntrna_database_path=replace_db_dir(_NTRNA_DATABASE_PATH.value),
        rfam_database_path=replace_db_dir(_RFAM_DATABASE_PATH.value),
        rna_central_database_path=replace_db_dir(
            _RNA_CENTRAL_DATABASE_PATH.value
        ),
        pdb_database_path=replace_db_dir(_PDB_DATABASE_PATH.value),
        seqres_database_path=replace_db_dir(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
    )
  else:
    print('Skipping running the data pipeline.')
    data_pipeline_config = None

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend='gpu')
    print(f'Found local devices: {devices}')

    print('Building model from scratch...')
    model_runner = ModelRunner(
        model_class=diffusion_model.Diffuser,
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
            ),ref_time_steps=_REF_TIME_STEPS.value
        ),
        device=devices[0],
        model_dir=pathlib.Path(_MODEL_DIR.value),
    )
  else:
    print('Skipping running model inference.')
    model_runner = None

  print(f'Processing {len(fold_inputs)} fold inputs.')
  for fold_input in fold_inputs:
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
        ref_pdb_path=_REF_PDB_PATH.value,
        ref_pkl_dump_path=_REF_PKL_DUMP_PATH.value,
        buckets=_BUCKETS
    )

  print(f'Done processing {len(fold_inputs)} fold inputs.')


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_dir',
  ])
  app.run(main)
