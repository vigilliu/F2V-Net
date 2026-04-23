import os
import SimpleITK as sitk
from tqdm import tqdm

# =========================
# 路径设置
# =========================
src_root = "/root/dof_project/3dct_point_dataset/data"
dst_root = "/root/dof_project/3dct_point_dataset_DS4/data"

scale = 4  # ↓ 1/4 downsample


# =========================
# 降采样函数
# =========================
def downsample_nii(input_path, output_path, scale=4):

    # 读取图像
    img = sitk.ReadImage(input_path)

    original_size = img.GetSize()      # (W,H,D)
    original_spacing = img.GetSpacing()

    # 新尺寸（1/4）
    new_size = [
        int(original_size[0] / scale),
        int(original_size[1] / scale),
        int(original_size[2] / scale),
    ]

    # spacing 必须扩大
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

    # CTA 用线性插值（不要nearest）
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_img = resampler.Execute(img)

    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    sitk.WriteImage(resampled_img, output_path)


# =========================
# 遍历数据集
# =========================
case_ids = sorted(os.listdir(src_root))

for case_id in tqdm(case_ids):

    input_path = os.path.join(
        src_root, case_id, "cta", f"{case_id}ct.nii.gz"
    )

    output_path = os.path.join(
        dst_root, case_id, "cta", f"{case_id}ct.nii.gz"
    )

    if not os.path.exists(input_path):
        continue

    downsample_nii(input_path, output_path, scale)

print("✅ Downsample finished!")