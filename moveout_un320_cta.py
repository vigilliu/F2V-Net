import os
import shutil
import nibabel as nib

DATA_ROOT = "./3dct_point_dataset/data"
POINT_ROOT = "./3dct_point_dataset/ctpoint_label"
BAD_ROOT = "./data_not_512_512_320"

EXPECTED_SHAPE = (512, 512, 320)

BAD_DATA_ROOT = os.path.join(BAD_ROOT, "data")
BAD_POINT_ROOT = os.path.join(BAD_ROOT, "ctpoint_label")

os.makedirs(BAD_DATA_ROOT, exist_ok=True)
os.makedirs(BAD_POINT_ROOT, exist_ok=True)

def move_case(case_id):
    src_data = os.path.join(DATA_ROOT, case_id)
    src_point = os.path.join(POINT_ROOT, case_id)

    dst_data = os.path.join(BAD_DATA_ROOT, case_id)
    dst_point = os.path.join(BAD_POINT_ROOT, case_id)

    if os.path.exists(src_data):
        shutil.move(src_data, dst_data)
        print(f"[MOVE] data/{case_id} -> {dst_data}")

    if os.path.exists(src_point):
        shutil.move(src_point, dst_point)
        print(f"[MOVE] ctpoint_label/{case_id} -> {dst_point}")
    else:
        print(f"[WARN] ctpoint_label/{case_id} 不存在")

bad_cases = []

for case_id in sorted(os.listdir(DATA_ROOT)):
    case_path = os.path.join(DATA_ROOT, case_id)
    if not os.path.isdir(case_path):
        continue

    cta_dir = os.path.join(case_path, "cta")
    if not os.path.isdir(cta_dir):
        print(f"[WARN] {case_id} 没有 cta 目录")
        continue

    for fname in os.listdir(cta_dir):
        if not fname.endswith(".nii.gz"):
            continue

        fpath = os.path.join(cta_dir, fname)
        img = nib.load(fpath)
        shape = img.shape

        if shape != EXPECTED_SHAPE:
            print(f"[BAD] {case_id}/{fname} shape={shape}")
            bad_cases.append(case_id)
        else:
            print(f"[OK]  {case_id}/{fname} shape={shape}")

# 去重后再移动（防止一个 case 多次触发）
bad_cases = sorted(set(bad_cases))

print("\n====== 开始移动异常 case ======")
for cid in bad_cases:
    move_case(cid)

print("\n🎯 完成：异常数据已移动至 data_not_512_512_320/")
