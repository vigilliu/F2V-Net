import os

root_data = "data"
root_label = "ctpoint_label"

start_id = 125
num_new = 10

for i in range(start_id, start_id + num_new):
    # data 目录
    os.makedirs(os.path.join(root_data, str(i), "cta"), exist_ok=True)
    os.makedirs(os.path.join(root_data, str(i), "tee"), exist_ok=True)

    # label 目录
    os.makedirs(os.path.join(root_label, str(i)), exist_ok=True)

print(f"Successfully created folders from {start_id} to {start_id + num_new - 1}.")
