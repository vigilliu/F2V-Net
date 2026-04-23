import os
import SimpleITK as sitk
from tqdm import tqdm

# =========================
# 路径
# =========================
src_root = "/root/dof_project/results"
dst_root = "/root/dof_project/results_DS4"

scale = 4


# =========================
# mask降采样
# =========================
def downsample_mask(input_path, output_path, scale=4):

    img = sitk.ReadImage(input_path)

    original_size = img.GetSize()      # (W,H,D)
    original_spacing = img.GetSpacing()

    # 新尺寸
    new_size = [
        int(original_size[0] / scale),
        int(original_size[1] / scale),
        int(original_size[2] / scale),
    ]

    # spacing扩大
    new_spacing = [
        original_spacing[0] * scale,
        original_spacing[1] * scale,
        original_spacing[2] * scale,
    ]

    # =========================
    # Resample
    # =========================
    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    # ⭐⭐⭐ mask必须最近邻
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(img)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(resampled, output_path)


# =========================
# 遍历case
# =========================
case_ids = sorted(os.listdir(src_root))

for case_id in tqdm(case_ids):

    input_path = os.path.join(
        src_root, case_id, "cta", f"{case_id}ct_seg.nii.gz"
    )

    output_path = os.path.join(
        dst_root, case_id, "cta", f"{case_id}ct_seg.nii.gz"
    )

    if not os.path.exists(input_path):
        continue

    downsample_mask(input_path, output_path, scale)

print("✅ Mask downsample finished!")