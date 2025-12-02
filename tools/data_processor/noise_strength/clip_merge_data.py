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

def load_clip_score_data(clip_score_dir):
    """加载clip score数据"""
    clip_score_data = {}
    
    for i in range(1,10):
        file_path = os.path.join(clip_score_dir, f'noise_strength_{i}.clipscore')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # 处理clip score文件结构：每个视频名称对应一个字典，包含prompt和score
                processed_data = {}
                for vid_name, vid_data in data.items():
                    if vid_name == 'avg_score':
                        # 保留avg_score
                        processed_data[vid_name] = vid_data
                    elif isinstance(vid_data, dict):
                        # 提取字典中的score值（假设只有一个键值对）
                        score_values = list(vid_data.values())
                        if score_values:
                            processed_data[vid_name] = score_values[0]  # 取第一个score值
                        else:
                            processed_data[vid_name] = ''
                    else:
                        processed_data[vid_name] = vid_data
                
                clip_score_data[f'noise_strength_{i}'] = processed_data
    
    return clip_score_data

def merge_data(motion_file, eval_file, clip_score_dir, output_file):
    """合并所有数据到CSV文件"""
    # 加载数据
    motion_data = load_motion_data(motion_file)
    eval_data = load_eval_data(eval_file)
    clip_score_data = load_clip_score_data(clip_score_dir)
    
    # 准备CSV数据
    csv_data = []
    
    # 创建表头
    header = ['原视频名称', '运动分数', '编辑后视频名称']
    for i in range(1,10):
        header.append(f'noise_strength_{i}_score')
    header.append('avg_score')
    header.append('prompt')
    
    # 处理每个评估记录
    for eval_record in eval_data:
        src_vid_name = eval_record['src_vid_name']
        vid_name = eval_record['vid_name']
        prompt = eval_record['prompt']

        scr_mp4_name = src_vid_name + '.mp4'
        
        # 获取运动分数
        motion_score = motion_data.get(scr_mp4_name, '')
        
        # 获取clip score数据
        row_data = [src_vid_name, motion_score, vid_name]
        
        # 添加每个noise strength的score
        for i in range(1,10):
            noise_key = f'noise_strength_{i}'
            if noise_key in clip_score_data:
                score_value = clip_score_data[noise_key].get(vid_name, '')
                row_data.append(score_value)
            else:
                row_data.append('')
        
        # 计算当前编辑后视频在不同noise strength下的平均score
        score_sum = 0
        score_count = 0
        for i in range(1,10):
            noise_key = f'noise_strength_{i}'
            if noise_key in clip_score_data and vid_name in clip_score_data[noise_key]:
                score_value = clip_score_data[noise_key][vid_name]
                if score_value != '' and score_value is not None:
                    score_sum += float(score_value)
                    score_count += 1
        
        # 计算平均分
        avg_score = score_sum / score_count if score_count > 0 else ''
        row_data.append(avg_score)
        
        # 添加prompt
        row_data.append(prompt)
        
        csv_data.append(row_data)
    
    avg_score_row = ['', '', 'avg_score']  # 前三列留空
    
    # 添加每个noise strength的avg_score
    for i in range(1,10):
        noise_key = f'noise_strength_{i}'
        if noise_key in clip_score_data and 'avg_score' in clip_score_data[noise_key]:
            avg_score_row.append(clip_score_data[noise_key]['avg_score'])
        else:
            avg_score_row.append('')
    
    # 添加两个空列（对应个人平均score和prompt列）
    avg_score_row.append('')  # 个人平均score列
    avg_score_row.append('')  # prompt列
    
    csv_data.append(avg_score_row)
    
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
    clip_score_dir = '/home/zrj/project/ori_v2v/streamv2v/tools/clip_score_log'
    output_file = '/home/zrj/project/ori_v2v/streamv2v/tools/statistics/noise_strength/clip_merged_data.csv'
    
    # 检查文件是否存在
    if not os.path.exists(motion_file):
        print(f"运动强度文件不存在：{motion_file}")
        return
    
    if not os.path.exists(eval_file):
        print(f"评估文件不存在：{eval_file}")
        return
    
    if not os.path.exists(clip_score_dir):
        print(f"clip score目录不存在：{clip_score_dir}")
        return
    
    # 合并数据
    merge_data(motion_file, eval_file, clip_score_dir, output_file)

if __name__ == '__main__':
    main()