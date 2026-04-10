import argparse
import sys
from src.data.raw_to_interim import main as step_raw_to_interim
from src.data.interim_to_processed import main as step_interim_to_processed
from src.data.import_external_videos import main as step_import_external

def run():
    parser = argparse.ArgumentParser(description="ASL Dataset Pipeline Controller")
    parser.add_argument(
        "--source", 
        choices=["raw", "external", "all"], 
        default="all",
        help="Chọn nguồn dữ liệu: 'raw' (từ webcam), 'external' (từ video), hoặc 'all' (cả hai)."
    )
    args = parser.parse_args()

    print("\n" + "="*50)
    print(f"BẮT ĐẦU PIPELINE - CHẾ ĐỘ: {args.source.upper()}")
    print("="*50)

    # Backup argv để tránh lỗi recursion khi gọi main() của các module khác
    original_argv = sys.argv

    # --- BƯỚC 1: THU THẬP DỮ LIỆU (DATA COLLECTION) ---
    # Chạy import video ngoại vi trước để đổ dữ liệu vào thư mục raw
    if args.source in ["external", "all"]:
        print("\n[STEP 1] Importing EXTERNAL VIDEOS to RAW...")
        step_import_external(override_args=[]) 

    # --- BƯỚC 2: TIỀN XỬ LÝ (PREPROCESSING) ---
    # Sau khi đã có đầy đủ file trong raw (cả webcam cũ và video mới import)
    # Chúng ta mới tiến hành chuẩn hóa (Normalization) toàn bộ sang Interim
    if args.source in ["raw", "external", "all"]:
        print("\n[STEP 2] Converting RAW to INTERIM (Normalization)...")
        sys.argv = [original_argv[0]] # Reset args cho module con
        step_raw_to_interim()

    # --- BƯỚC 3: ĐÓNG GÓI (PACKAGING) ---
    # Gom tất cả từ Interim -> Processed (.npz) để sẵn sàng cho Training
    print("\n[STEP 3] Aggregating all INTERIM data into TRAIN.NPZ...")
    sys.argv = [original_argv[0]]
    step_interim_to_processed()

    print("\n" + "="*50)
    print("PIPELINE HOÀN TẤT!")
    print("="*50)