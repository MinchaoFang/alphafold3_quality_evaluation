import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.rosetta.protocols.comparative_modeling import *
from pyrosetta.rosetta.protocols.simple_moves import *

def initialize_pyrosetta():
    """初始化PyRosetta"""
    init_options = "-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -ignore_waters -mute all"
    pyrosetta.init(init_options)

def parse_sequences(sequence_string):
    """解析多链序列"""
    return sequence_string.split(':')

def get_chain_bounds(pose, chain_id='A'):
    """获取指定链的范围"""
    start_res = 1
    end_res = 0
    
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) == chain_id:
            if end_res == 0:
                start_res = i
            end_res = i
    
    if end_res == 0:
        raise ValueError(f"在结构中没有找到Chain {chain_id}")
        
    return start_res, end_res

def mutate_chain_sequence(pose, new_sequence, start_res, end_res):
    """
    将新序列赋给指定链区域
    
    参数:
    pose: PyRosetta pose对象
    new_sequence: 新的氨基酸序列
    start_res: 起始残基编号
    end_res: 结束残基编号
    """
    mutater = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    
    for i, aa in enumerate(new_sequence):
        pose_position = start_res + i
        if pose_position > end_res:
            break
            
        # 转换单字母氨基酸代码为PyRosetta名称
        aa_name = pyrosetta.rosetta.core.chemical.aa_from_oneletter_code(aa)
        
        # 设置突变
        mutater.set_res_name(aa_name)
        mutater.set_target(pose_position)
        mutater.apply(pose)

def build_full_atom_model(pdb_file, sequence_string):
    """重建完整原子模型"""
    # 初始化PyRosetta
    initialize_pyrosetta()
    
    # 解析序列
    chains = parse_sequences(sequence_string)
    chain_a_seq = chains[0]
    
    # 读取骨架结构
    pose = pyrosetta.pose_from_pdb(pdb_file)
    
    # 获取Chain A的范围
    chain_a_start, chain_a_end = get_chain_bounds(pose, 'A')
    chain_a_length = chain_a_end - chain_a_start + 1
    
    # 验证Chain A序列长度
    if chain_a_length != len(chain_a_seq):
        raise ValueError(f"Chain A序列长度({len(chain_a_seq)})与结构中的残基数({chain_a_length})不匹配")
    
    # 替换Chain A序列
    mutate_chain_sequence(pose, chain_a_seq, chain_a_start, chain_a_end)
    
    # 重建侧链
    rebuild_chain_a_sidechains(pose, chain_a_start, chain_a_end)
    
    return pose, chain_a_start, chain_a_end

def rebuild_chain_a_sidechains(pose, start_res, end_res):
    """重建Chain A侧链"""
    # 创建task factory
    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
    
    # 基础操作
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    
    # 创建只允许重建Chain A的操作
    prevent_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepacking()
    for i in range(1, pose.total_residue() + 1):
        if i < start_res or i > end_res:  # 如果不是Chain A的残基
            prevent_repacking.include_residue(i)
    task_factory.push_back(prevent_repacking)
    
    # 设置打包选项
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(task_factory)
    
    # 执行侧链重建
    packer.apply(pose)

def optimize_chain_a(pose, start_res, end_res):
    """优化Chain A结构"""
    # 创建评分函数
    scorefxn = pyrosetta.get_fa_scorefxn()
    
    # 设置最小化选项，只针对Chain A
    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    for i in range(1, pose.total_residue() + 1):
        if start_res <= i <= end_res:
            movemap.set_chi(i, True)
            movemap.set_bb(i, False)
    
    # 创建最小化器
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        movemap,
        scorefxn,
        'lbfgs_armijo_nonmonotone',
        0.001,
        True,
        False
    )
    
    # 执行能量最小化
    min_mover.apply(pose)


def process(pdb_file, sequence_string, output_file):
    """主函数"""
    try:
        # 构建模型
        pose, chain_a_start, chain_a_end = build_full_atom_model(pdb_file, sequence_string)
        
        ## 优化Chain A
        #optimize_chain_a(pose, chain_a_start, chain_a_end)
        
        ## 验证Chain A
        #validation_results = validate_chain_a(pose, chain_a_start, chain_a_end)
        #print("Chain A验证结果:", validation_results)
        
        # 保存结果
        pose.dump_pdb(output_file)
        print(f"模型已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise


import pandas as pd
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process PDB files with sequences from CSV')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default='output_models',
                       help='Directory for output files (default: output_models)')
    parser.add_argument('--begin_row', type=int, default=0,
                       help='Starting row index (default: 0)')
    parser.add_argument('--end_row', type=int, default=None,
                       help='Ending row index (default: process all rows)')
    
    return parser.parse_args()

def process_pdb_files(args):
    # 读取CSV文件
    df = pd.read_csv(args.csv_path)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定处理范围
    end_row = args.end_row if args.end_row is not None else len(df)
    df_subset = df.iloc[args.begin_row:end_row]
    
    # 遍历处理每一行
    for index, row in df_subset.iterrows():
        pdb_file = row['scaffold_path']
        sequence_string = row['mpnn_sequence']
        output_file = os.path.join(
            args.output_dir,
            f"{os.path.basename(row['esmfold_path']).replace('esmfold_','').replace('.pdb','')}.pdb"
        )
        
        print(f"处理第 {index + 1} 行：PDB 文件: {pdb_file}, 序列: {sequence_string}")
        
        try:
            process(pdb_file, sequence_string, output_file)
        except Exception as e:
            print(f"第 {index + 1} 行处理出错: {e}")

def main():
    args = parse_arguments()
    process_pdb_files(args)

if __name__ == "__main__":
    main()

