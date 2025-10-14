import os
import torch
from torchvision import transforms as T
import numpy as np
import random  # 亂數控制
import argparse  # 命令列參數處理
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import torchvision.transforms as transforms
import cv2
from PIL import Image # 雖然 transform 用到了，但直接用 cv2 讀寫更一致

def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋


# =======================
# Utilities
# =======================
def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1  # 沒有GPU可用

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        # memory_reserved = torch.cuda.memory_reserved(i) # 這個在某些情況下會顯示較高，我們更關注已分配的
        gpu_memory.append((i, memory_allocated)) # 只用 allocated

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


def visualize_and_save(original_img_rgb, recon_img_rgb, anomaly_map_normalized, binary_mask,
                       save_path_base):
    """
    將推論結果可視化並儲存成圖片。

    Args:
        original_img_rgb (np.ndarray): 原始輸入影像 (H, W, 3)，RGB格式，值域 [0, 255]。
        recon_img_rgb (np.ndarray): 重建後的影像 (H, W, 3)，RGB格式，值域 [0, 255]。
        anomaly_map_normalized (np.ndarray): 異常分數圖 (H, W)，值域 [0, 255]，8-bit 整數。
        binary_mask (np.ndarray): 二值化的異常遮罩 (H, W)，值為 0 或 255。
        save_path_base (str): 儲存檔案的基礎路徑與檔名 (不含副檔名)。
    """
    # 確保儲存目錄存在
    save_dir = os.path.dirname(save_path_base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 將 anomaly_map 轉換為熱力圖
    heatmap_color = cv2.applyColorMap(anomaly_map_normalized, cv2.COLORMAP_JET)

    # 將熱力圖疊加到原始影像上
    # 因為 original_img_rgb 是 RGB，而 cv2.addWeighted 期望 BGR，所以先轉換一下
    original_img_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_img_bgr, 0.6, heatmap_color, 0.4, 0)

    # 將二值化遮罩轉為三通道，方便合併 (BGR 格式)
    binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR) # 注意 binary_mask 已經是 [0, 255]

    # 將四張圖拼接成一張大圖 (原始圖 | 重建圖 | 疊加熱力圖 | 二值圖)
    # recon_img_rgb 也是 RGB，需要轉 BGR
    combined_img = np.hstack([original_img_bgr, cv2.cvtColor(recon_img_rgb, cv2.COLOR_RGB2BGR), overlay, binary_mask_color])

    # 儲存合併後的影像
    cv2.imwrite(f"{save_path_base}_results.png", combined_img)
    print(f"✅ 結果已儲存至: {save_path_base}_results.png")


# --- 修改後的 run_inference 函數 ---
def run_inference(img_path, student_model,student_seg_model, device, save_path_base, img_dim=256):
    # 1. 圖像預處理
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        # 如果你的訓練資料有正規化到 [-1, 1]，請在這裡加上 Normalize
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入原始圖像並轉為 RGB
    image = Image.open(img_path).convert("RGB") 
    
    # 儲存原始圖像，以便 visualize_and_save 使用 (resize 到 img_dim x img_dim)
    # 從 PIL Image 轉為 numpy array, 並且值域為 [0, 255]
    original_img_resized_pil = image.resize((img_dim, img_dim), Image.LANCZOS)
    original_img_rgb_np = np.array(original_img_resized_pil) # 值域 [0, 255]

    # 將圖像轉換為模型輸入張量
    input_tensor = transform(image).unsqueeze(0).to(device) # 添加批次維度並移到GPU

    with torch.no_grad():
        # 2. 將圖像輸入到學生模型的重建子網路
        student_recon_output_tensor = student_model(image)

        # 3. 將重建輸出和原始輸入圖像級聯
        joined_input_for_discriminator = torch.cat((student_recon_output_tensor.detach(), image), dim=1)

        # 4. 將級聯輸入傳遞給學生模型的判別子網路
        student_seg_logits = student_seg_model(joined_input_for_discriminator)

        # 5. 處理分割輸出 (Softmax)
        student_seg_map_sm = torch.softmax(student_seg_logits, dim=1)
        # 提取異常通道 (假設是通道 1)
        anomaly_map_raw = student_seg_map_sm[0, 1, :, :].cpu().numpy() # 原始值域 [0, 1]

        # 將重建圖像張量轉換為 NumPy 陣列，值域 [0, 255]
        # (C, H, W) -> (H, W, C), 然後從 [0, 1] 縮放到 [0, 255] 並轉為 uint8
        recon_image_np = (student_recon_output_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # 將 anomaly_map_raw (值域 [0, 1]) 歸一化到 [0, 255]
        anomaly_map_normalized_uint8 = (anomaly_map_raw * 255).astype(np.uint8)

        # 如果需要二值化遮罩，可以設定一個閾值
        threshold = 0.5 # 可以調整閾值
        binary_mask = (anomaly_map_raw > threshold).astype(np.uint8) * 255 # 0或255

        # 調用可視化函數
        visualize_and_save(original_img_rgb_np, recon_image_np,
                           anomaly_map_normalized_uint8, binary_mask, save_path_base)

    return anomaly_map_raw, binary_mask # 返回原始的 float 異常圖和二值遮罩 (方便後續指標計算)


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 建立主存檔資料夾
    save_root = "./inference_results" # 推理結果通常保存在不同的目錄
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print("🔄 開始測試，共有物件類別:", len(obj_names))

    IMG_CHANNELS = 3
    IMG_SEG_CHANNELS = 6
    IMG_SEG_CHANNELS_OUT = 2
    for obj_name in obj_names:
        print(f"▶️測試物件類別: {obj_name}")
        student_model = ReconstructiveSubNetwork(
            in_channels=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            base_width=64,  # 學生重建網路較窄
        ).to(device)

        student_seg_model = DiscriminativeSubNetwork(
            in_channels=IMG_SEG_CHANNELS,
            out_channels=IMG_SEG_CHANNELS_OUT,
            base_channels=32,  # 學生重建網路較窄
            out_features=False,
        ).to(device)

        # 載入訓練好的學生模型權重
        # 建議載入基於 AUROC 保存的最佳模型
        # model_weights_path = os.path.join(args.checkpoint_dir, f"{obj_name}.pckl") # ⬅️ 確保這裡的路徑正確
        model_best_recon_weights_path = './student_model_checkpoints/bottle_best_recon.pckl'  # ⬅️ 我的的權重路徑
        if not os.path.exists(model_best_recon_weights_path):
            print(f"❌ 錯誤: 未找到模型權重檔案: {model_best_recon_weights_path}，請檢查路徑或訓練是否完成。")
            continue
            
        student_model.load_state_dict(
            torch.load(model_best_recon_weights_path, map_location=device))
        
        model_best_seg_weights_path = './student_model_checkpoints/bottle_best_seg.pckl'  # ⬅️ 我的的權重路徑
        if not os.path.exists(model_best_seg_weights_path):
            print(f"❌ 錯誤: 未找到模型權重檔案: {model_best_seg_weights_path}，請檢查路徑或訓練是否完成。")
            continue
            
        student_seg_model.load_state_dict(
            torch.load(model_best_seg_weights_path, map_location=device))

        # --- 2. 設定為評估模式 ---
        student_seg_model.eval()

        test_path = os.path.join(args.mvtec_root, obj_name, 'test')  # 測試資料路徑
        items = ['good', 'broken_large', 'broken_small',
                 'contamination']  # 測試資料標籤
        print(f"🔍 測試資料夾：{test_path}，共 {len(items)} 類別")

        # 依類別逐張讀取影像並執行推論
        for item in items:
            item_path = os.path.join(test_path, item)
            # 建立該類別的輸出資料夾
            output_dir = os.path.join(save_root, obj_name, item)
            os.makedirs(output_dir, exist_ok=True)  # 確保資料夾存在

            if not os.path.exists(item_path):
                print(f"⚠️ 警告: 路徑不存在 {item_path}，跳過。")
                continue

            img_files = [
                f for f in os.listdir(item_path)
                if f.endswith('.png') or f.endswith('.jpg')
            ]

            print(f"\n📂 類別：{item}，共 {len(img_files)} 張影像")

            for img_name in img_files:
                img_path = os.path.join(item_path, img_name)
                print(f"🖼️ 處理影像：{img_path}")

                # 去掉副檔名，只取檔名主體
                base_name, _ = os.path.splitext(img_name)
                # 設定儲存路徑
                save_path_base = os.path.join(output_dir, base_name)

                # --- 執行推理 ---
                anomaly_map, binary_mask = run_inference(
                    img_path, student_model,student_seg_model, device, save_path_base)
        print(f"\n✅ 物件類別 {obj_name} 測試完成！")
    print("\n🎉 所有測試已完成！")


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')
    parser.add_argument('--mvtec_root', type=str, default='./mvtec', help='Path to the MVTec dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./save_files', help='Directory to load model checkpoints')
    
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:
        # 使用CPU
        main(picked_classes, args)
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)