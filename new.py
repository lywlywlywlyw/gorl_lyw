import h5py

file_path = '/root/GoRL/datasets/d4rl/walker2d-medium-expert-v2.hdf5'

with h5py.File(file_path, 'r') as f:
    print("文件结构:")
    for key in f.keys():
        print(f"  {key}: {type(f[key])}")
        if isinstance(f[key], h5py.Dataset):
            print(f"    shape: {f[key].shape}")
    
    # 如果是robomimic格式，通常demo数量在 'data' 组里
    if 'data' in f:
        print(f"\n演示数量: {len(f['data'].keys())}")
        # 打印每个演示的key
        for i, demo_key in enumerate(f['data'].keys()):
            print(f"  Demo {i+1}: {demo_key}")