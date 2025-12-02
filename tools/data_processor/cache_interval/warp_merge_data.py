import json
import csv
import os
import glob

def load_motion_data(motion_file):
    """加载运动强度数据"""
    motion_data = {}
    with open(motion_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                video_name, motion_score = row
                motion_data[video_name] = float(motion_score)
    return motion_data

def load_eval_data(eval_file):
    """加载评估数据"""
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    return eval_data

def load_warp_error_data(warp_error_dir):
    """加载warp error数据"""
    warp_error_data = {}
    
    for i in range(1,9):
        file_path = os.path.join(warp_error_dir, f'cache_interval_{i}.warperror')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                warp_error_data[f'cache_interval_{i}'] = data
    
    return warp_error_data

def merge_data(motion_file, eval_file, warp_error_dir, output_file):
    """合并所有数据到CSV文件"""
    # 加载数据
    motion_data = load_motion_data(motion_file)
    eval_data = load_eval_data(eval_file)
    warp_error_data = load_warp_error_data(warp_error_dir)
    
    # 准备CSV数据
    csv_data = []
    
    # 创建表头
    header = ['原视频名称', '运动分数', '编辑后视频名称']
    for i in range(1,9):
        header.append(f'cache_interval_{i}_error')
    header.append('avg_error')
    header.append('prompt')
    
    # 处理每个评估记录
    for eval_record in eval_data:
        src_vid_name = eval_record['src_vid_name']
        vid_name = eval_record['vid_name']
        prompt = eval_record['prompt']

        scr_mp4_name = src_vid_name + '.mp4'
        
        # 获取运动分数
        motion_score = motion_data.get(scr_mp4_name, '')
        
        # 获取warp error数据
        row_data = [src_vid_name, motion_score, vid_name]
        
        # 添加每个noise strength的error
        for i in range(1,9):
            noise_key = f'cache_interval_{i}'
            if noise_key in warp_error_data:
                error_value = warp_error_data[noise_key].get(vid_name, '')
                row_data.append(error_value)
            else:
                row_data.append('')
        
        # 计算当前编辑后视频在不同noise strength下的平均error
        error_sum = 0
        error_count = 0
        for i in range(1,9):
            noise_key = f'cache_interval_{i}'
            if noise_key in warp_error_data and vid_name in warp_error_data[noise_key]:
                error_value = warp_error_data[noise_key][vid_name]
                if error_value != '' and error_value is not None:
                    error_sum += float(error_value)
                    error_count += 1
        
        # 计算平均分
        avg_error = error_sum / error_count if error_count > 0 else ''
        row_data.append(avg_error)
        
        # 添加prompt
        row_data.append(prompt)
        
        csv_data.append(row_data)
    
    # 添加avg_error行（每个noise strength下的总体平均error）
    avg_error_row = ['', '', 'avg_error']  # 前三列留空
    
    # 添加每个noise strength的avg_error
    for i in range(1,9):
        noise_key = f'cache_interval_{i}'
        if noise_key in warp_error_data and 'avg_error' in warp_error_data[noise_key]:
            avg_error_row.append(warp_error_data[noise_key]['avg_error'])
        else:
            avg_error_row.append('')
    
    # 添加两个空列（对应个人平均error和prompt列）
    avg_error_row.append('')  # 个人平均error列
    avg_error_row.append('')  # prompt列
    
    csv_data.append(avg_error_row)
    
    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)
    
    print(f"数据合并完成！输出文件：{output_file}")
    print(f"共处理 {len(csv_data)} 条记录")

def main():
    # 定义文件路径
    motion_file = '/home/zrj/project/ori_v2v/streamv2v/tools/motion_strengths2.csv'
    eval_file = '/home/zrj/project/ori_v2v/streamv2v/tools/user_study_upload/eval.json'
    warp_error_dir = '/home/zrj/project/ori_v2v/streamv2v/tools/warp_error_log'
    output_file = '/home/zrj/project/ori_v2v/streamv2v/tools/statistics/cache_interval/warp_merged_data.csv'
    
    # 检查文件是否存在
    if not os.path.exists(motion_file):
        print(f"运动强度文件不存在：{motion_file}")
        return
    
    if not os.path.exists(eval_file):
        print(f"评估文件不存在：{eval_file}")
        return
    
    if not os.path.exists(warp_error_dir):
        print(f"warp error目录不存在：{warp_error_dir}")
        return
    
    # 合并数据
    merge_data(motion_file, eval_file, warp_error_dir, output_file)

if __name__ == '__main__':
    main()