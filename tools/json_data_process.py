import json

# -------------------------- 配置参数（根据实际情况修改）--------------------------
INPUT_JSON_PATH = "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/eval.json"    # 原始 JSON 文件路径（请替换为你的文件路径）
OUTPUT_JSON_PATH = "/home/zrj/project/ori_v2v/streamv2v/vid2vid/source_video/eval2.json" # 输出 JSON 文件路径（可自定义）
FILE_PATH = "./source_video"         # 固定的 file_path 值
DIFFUSION_STEPS = "4"                # 固定的 diffusion_steps 值
NOISE_STRENGTH = "0.4"               # 固定的 noise_strength 值
MODEL_ID = "/home/zrj/project/ori_v2v/streamv2v/data/checkpoints/stable-diffusion-1.5"  # 固定 model_id
# --------------------------------------------------------------------------------

def convert_json():
    # 1. 读取原始 JSON 文件
    try:
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            original_data = json.load(f)  # 原始数据是列表形式
        print(f"成功读取原始文件：{INPUT_JSON_PATH}，共 {len(original_data)} 条数据")
    except FileNotFoundError:
        print(f"错误：未找到原始文件 {INPUT_JSON_PATH}，请检查文件路径")
        return
    except json.JSONDecodeError:
        print(f"错误：原始文件 {INPUT_JSON_PATH} 不是合法的 JSON 格式")
        return

    # 2. 转换数据格式（包含所有要求字段：file_path、src_vid_name、vid_name、prompt、model_id、diffusion_steps、noise_strength）
    converted_data = []
    for item in original_data:
        new_item = {
            "file_path": FILE_PATH,
            "src_vid_name": item["src_vid_name"],  # 原始 src_vid_name
            "vid_name": item["vid_name"],          # 原始 vid_name
            "prompt": item["prompt"],              # 原始 prompt
            "model_id": MODEL_ID,                  # 新增：固定 model_id
            "diffusion_steps": DIFFUSION_STEPS,    # 固定值
            "noise_strength": NOISE_STRENGTH       # 固定值
        }
        converted_data.append(new_item)

    # 3. 写入 JSON Lines 文件（每行一个 JSON 对象）
    try:
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            for data in converted_data:
                # 将每个 JSON 对象转为字符串并写入一行（ensure_ascii=False 保留中文）
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")  # 换行分隔
        print(f"成功生成目标文件：{OUTPUT_JSON_PATH}")
        print(f"转换完成！共生成 {len(converted_data)} 条记录")
        print(f"每条记录包含字段：file_path、src_vid_name、vid_name、prompt、model_id、diffusion_steps、noise_strength")
    except Exception as e:
        print(f"错误：写入文件失败 - {str(e)}")

if __name__ == "__main__":
    convert_json()