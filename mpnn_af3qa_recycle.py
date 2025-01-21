import os
import argparse
import pandas as pd
import sys
from pathlib import Path
import json
import subprocess
import shutil
from Bio.PDB import MMCIFParser, PDBParser, PDBIO
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import os
import shutil
from input_pkl_preprocess import process_single_file
import subprocess
import math
import json
from copy import deepcopy
from Bio.PDB import PDBParser, MMCIFParser
import json
import csv
import os
import sys
sys.path.insert(0,"/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching")
from omegaconf import OmegaConf
from colabdesign.af.model import mk_af_model
from ProteinMPNN.protein_mpnn_utils import model_init
from ProteinMPNN.protein_mpnn_pyrosetta import mpnn_design
from data.parsers import from_pdb_string
from scripts.self_consistency_evaluation import run_folding_and_evaluation
from data import protein
from Bio.PDB import MMCIFParser, PDBIO
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF2 and AF3')
    parser.add_argument('--pdb_list', type=str, required=True,
                       help='Path to text file containing list of PDB files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing PDB files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--num_recycles', type=int, default=10,
                       help='Number of recycles to perform')
    parser.add_argument('--template_path', type=str, required=True,
                       help='Path to template.json file')
    return parser.parse_args()

def convert_cif_to_pdb(cif_file, pdb_file):
    """将CIF文件转换为PDB文件"""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)
        return True
    except Exception as e:
        print(f"CIF转PDB失败: {str(e)}")
        return False

def self_consistency_init():
    mpnn_config_dict = {
            "ca_only": True,
            "model_name": "v_48_020",
            'backbone_noise': 0.00,
            'temperature': 0.1,
            'num_seqs': 4,
        }
    mpnn_config_dict = OmegaConf.create(mpnn_config_dict)
    mpnn_model = model_init(mpnn_config_dict, device='cuda')
    cfg = OmegaConf.load('/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching/configs/inference.yaml')
    af2_configs = cfg.inference.self_consistency.structure_prediction.alphafold
    af2_setting = {
        "models": [3] ,
        "num_recycles": af2_configs.num_recycles,
        'prefix': 'monomer',
        'params_dir': f'/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching/{cfg.inference.self_consistency.structure_prediction.alphafold.params_dir}'
    }
    prediction_model = mk_af_model(
                protocol="hallucination", 
                initial_guess=False, 
                use_initial_atom_pos=False, num_recycles=af2_configs.num_recycles, 
                data_dir=af2_setting['params_dir'],
            )
    return mpnn_model, mpnn_config_dict, prediction_model, af2_setting

def get_chain_sequence(file_path: str, chain_id: str) -> str:
    """从PDB文件中获取指定链的氨基酸序列"""
    try:
        amino_acid_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        standard_amino_acids = set(amino_acid_map.keys())
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", file_path)
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                sequence = ""
                for residue in chain:
                    if residue.get_resname() in standard_amino_acids:
                        sequence += amino_acid_map[residue.get_resname()]
                return sequence
        return None
    except Exception as e:
        print(f"获取序列失败: {str(e)}")
        return None

def calculate_average_b_factor(cif_file: str) -> float:
    """计算CIF文件中的平均B-factor"""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        
        b_factors = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        b_factors.append(atom.bfactor)
        
        return sum(b_factors) / len(b_factors) if b_factors else 0.0
    except Exception as e:
        print(f"计算B-factor失败: {str(e)}")
        return 0.0

def process_confidence_metrics(confidence_json_path: str, cif_path: str) -> Dict:
    """处理AF3输出的置信度指标"""
    try:
        with open(confidence_json_path, 'r') as f:
            confidence_json = json.load(f)
        
        ptm = confidence_json['chain_ptm'][0]
        chain_pair_iptm = confidence_json['chain_pair_iptm'][0][1:]
        iptm = sum(chain_pair_iptm) / len(chain_pair_iptm) if chain_pair_iptm else 0
        
        chain_pair_pae_min = confidence_json['chain_pair_pae_min'][0][1:]
        ipae = sum(chain_pair_pae_min) / len(chain_pair_pae_min) if chain_pair_pae_min else 0
        
        plddt = calculate_average_b_factor(cif_path)
        
        return {
            'AF3_PTM': ptm,
            'AF3_iPTM': iptm,
            'AF3_iPAE': ipae,
            'AF3_pLDDT': plddt,
            'AF3_Status': 'Success'
        }
    except Exception as e:
        print(f"处理AF3指标失败: {str(e)}")
        return {
            'AF3_PTM': np.nan,
            'AF3_iPTM': np.nan,
            'AF3_iPAE': np.nan,
            'AF3_pLDDT': np.nan,
            'AF3_Status': 'Failed'
        }

def self_consistency(scaffold_path,output_path_tag, mpnn_model, mpnn_config_dict, prediction_model, af2_setting):
    
    mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict,
                protein_path=scaffold_path,
                model=mpnn_model,
                design_chains=['A']
            )
    #print(mpnn_seqs)
    import pandas as pd
    import re

    # 假设 mpnn_seqs, scaffold_path, prediction_model 等变量已定义
    results_list = []
    for i in range(len(mpnn_seqs)):
        sequence = mpnn_seqs[i].split(':')[0] if ':' in mpnn_seqs[i] else mpnn_seqs[i]
        sequence = re.sub("[^A-Z]", "", sequence.upper())
        scaffold_prot = from_pdb_string(open(scaffold_path).read(), 'A' )
        evaluated_results, pred_prots = run_folding_and_evaluation(prediction_model, sequence, scaffold_prot, None, af2_setting, template_chains=None)
        #print(evaluated_results)
        # 添加额外信息（如序列编号或序列本身）
        for result in evaluated_results:
            result['sequence'] = sequence  # 保存序列
            result['index'] = i           # 保存当前循环索引
            results_list.append(result)
    for j, (result, pred_prot) in enumerate(zip(evaluated_results, pred_prots)):
        fold_path = os.path.join(os.path.dirname(scaffold_path), output_path_tag + f"_af2_{j}.pdb")
        with open(fold_path, 'w') as f:
            f.write(protein.to_pdb(pred_prot))
        result["alphafold_path"] = fold_path
        result['mpnn_sequence'] = mpnn_seqs[i]
   
    return results_list

import subprocess

def run_alphafold3(json_path: str, pkl_path: str, output_dir: str) -> bool:
    """运行AlphaFold3命令"""
    try:
        command = [
            "module", "load", "alphafold/3_a40-tmp", "&&",
            "singularity", "exec",
            "--nv",
            "--bind", f"{output_dir}:/root/output",
            "--bind", "/storage/caolongxingLab/fangminchao/",
            "--bind", "/storage/caolongxingLab/share/",
            "--bind", "/storage/caolongxingLab/fangminchao/tools/alphafold3/model:/root/models",
            "--bind", "/storage/caolongxingLab/fangminchao/database/AF3/public_databases:/root/public_databases",
            "/soft/bio/alphafold/3/alphafold3.sif",
            "python", "/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/run_alphafold_avail.py",
            "--json_path="+json_path,
            "--ref_pdb_path="+pkl_path,
            "--ref_time_steps=50",
            "--model_dir=/root/models",
            "--db_dir=/root/public_databases",
            "--output_dir=/root/output",
            "--jax_compilation_cache_dir=/root/output/jax_compilation",
            "--norun_data_pipeline"
        ]
        
        result = subprocess.run(
            " ".join(command),
            shell=True,
            text=True,
            capture_output=True,
            timeout=3600  # 1小时超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("AF3运行超时")
        return False
    except Exception as e:
        print(f"AF3运行错误: {str(e)}")
        return False


def process_single_pdb(pdb_file: str, 
                      cycle: int,
                      input_dir: str,
                      output_dir: str,
                      template_path: str,
                      mpnn_model,
                      mpnn_config_dict,
                      prediction_model,
                      af2_setting) -> Dict:
    """处理单个PDB文件的一个循环"""
    metrics = {
        'PDB': pdb_file,
        'Cycle': cycle + 1,
        'AF2_RMSD': np.nan,
        'AF2_pLDDT': np.nan,
        'AF3_PTM': np.nan,
        'AF3_iPTM': np.nan,
        'AF3_iPAE': np.nan,
        'AF3_pLDDT': np.nan,
        'AF3_Status': 'Not Run'
    }
    
    try:
        # 设置目录
        target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制和重命名文件
        source_file = os.path.join(input_dir, pdb_file) if cycle == 0 else input_dir
        copied_file = os.path.join(target_dir, pdb_file.replace(".pdb", f"_recycle_{cycle+1}.pdb"))

        shutil.copy(source_file, copied_file)
        
        # 运行AF2一致性检查
        results_af2 = self_consistency(
            copied_file, 
            pdb_file.replace(".pdb", f"_recycle_{cycle+1}"),
            mpnn_model, 
            mpnn_config_dict, 
            prediction_model, 
            af2_setting
        )
        
        # 获取最小RMSD和对应的pLDDT
        min_rmsd = float('inf')
        corresponding_plddt = None
        for entry in results_af2:
            if entry['rmsd'] < min_rmsd:
                min_rmsd = entry['rmsd']
                corresponding_plddt = entry['plddt']
        
        metrics['AF2_RMSD'] = min_rmsd
        metrics['AF2_pLDDT'] = corresponding_plddt
        
        # 准备运行AF3
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        tag = f"{pdb_file}_recycle_{cycle+1}"
        json_path = os.path.join(target_dir, pdb_file.replace(".pdb", ".json"))
        
        input_json = template.copy()
        input_json['name'] = tag.replace(".pdb", "")
        input_json['sequences'][0]['protein']["sequence"] = get_chain_sequence(copied_file, 'A')
        
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process
        result, file_name, error = process_single_file(( copied_file, target_dir, target_dir))
        
        # 运行AF3
        pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
        success = run_alphafold3(json_path, pkl_path, target_dir)
        
        if success:
            # 获取AF3输出路径
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            confidence_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                         tag.replace(".pdb", "") + "_summary_confidences.json")
            
            # 获取AF3指标
            af3_metrics = process_confidence_metrics(confidence_path, cif_path)
            metrics.update(af3_metrics)
            
            # 转换CIF到PDB用于下一轮循环
            pdb_output = cif_path.replace(".cif", ".pdb")
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output
            else:
                print(f"CIF转换PDB失败，使用原始文件继续")
                return metrics, copied_file
        else:
            print(f"AF3处理失败，使用原始文件继续")
            metrics['AF3_Status'] = 'Failed'
            return metrics, copied_file
            
    except Exception as e:
        print(f"处理文件失败: {str(e)}")
        return metrics, copied_file

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template file {args.template_path} not found!")
    
    # 初始化模型
    mpnn_model, mpnn_config_dict, prediction_model, af2_setting = self_consistency_init()
    
    # 读取PDB列表
    with open(args.pdb_list, 'r') as f:
        pdb_files = [line.strip() for line in f if line.strip()]
    
    # 处理每个PDB文件
    all_results = []
    for pdb_file in pdb_files:
        print(f"\nProcessing {pdb_file}...")
        current_input = args.input_dir
        print()
        for cycle in range(args.num_recycles):
            print(f"  Starting cycle {cycle+1}")
            try:
                metrics, next_input = process_single_pdb(
                    pdb_file,
                    cycle,
                    current_input,  # 使用当前输入文件路径
                    args.output_dir,
                    args.template_path,
                    mpnn_model,
                    mpnn_config_dict,
                    prediction_model,
                    af2_setting
                )
                all_results.append(metrics)
                current_input = next_input  # 更新下一轮的输入文件
                
                # 保存每轮的结果
                df = pd.DataFrame(all_results)
                csv_path = os.path.join(args.output_dir, 'processing_results.csv')
                df.to_csv(csv_path, index=False)
                
                print(f"  Cycle {cycle+1} completed:")
                print(f"    AF2 RMSD: {metrics['AF2_RMSD']:.3f}")
                print(f"    AF2 pLDDT: {metrics['AF2_pLDDT']:.3f}")
                print(f"    AF3 Status: {metrics['AF3_Status']}")
                if metrics['AF3_Status'] == 'Success':
                    print(f"    AF3 PTM: {metrics['AF3_PTM']:.3f}")
                    print(f"    AF3 pLDDT: {metrics['AF3_pLDDT']:.3f}")
                
            except Exception as e:
                print(f"  Error in cycle {cycle+1}: {str(e)}")
                continue
    
    # 最终保存所有结果
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'processing_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    main()