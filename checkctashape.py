import os
import nibabel as nib

DATA_ROOT = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/data"
EXPECTED_SHAPE = (512, 512, 320)

wrong_cases = []

total_cta = 0
correct_cta = 0

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

        total_cta += 1

        fpath = os.path.join(cta_dir, fname)
        img = nib.load(fpath)
        shape = img.shape

        if shape != EXPECTED_SHAPE:
            wrong_cases.append((case_id, fname, shape))
        else:
            correct_cta += 1
            print(f"[OK] {case_id}/{fname} shape={shape}")

print("\n====== 检查结果 ======")
print(f"CTA 总数       : {total_cta}")
print(f"符合尺寸数量   : {correct_cta}")
print(f"不符合尺寸数量 : {total_cta - correct_cta}")

if len(wrong_cases) == 0:
    print("🎉 所有 CTA 尺寸均为 (512, 512, 320)")
else:
    print("❌ 以下 CTA 尺寸不符合要求：")
    for cid, fname, shape in wrong_cases:
        print(f"  {cid}/{fname} -> {shape}")
