import torch
from pathlib import Path
from src.models.model import GastroNet
from src.server import GastroServer
import sys
import argparse

'''
해당 파일은 메인 서버에서 돌리는 코드입니다.
실행시 받은 가중치들은 한 번에 모아서 평균을 내고 해당 가중치를 저장(덮어쓰움)합니다.
나온 가중치를 각 client에 보내면 됩니다.

실행 예시)
uploads/round_1 : 가중치 폴더 위치
.models/global_v1.pth : 합산된 가중치를 저장하고 싶은 폴더위치와 파일 이름
python -m src.run_server uploads/round_1 --output ./models/global_v1.pth
'''

def run_aggregation(weights_dir, output_path):
    target_dir = Path(weights_dir)
    
    if not target_dir.exists():
        sys.exit(f"{weights_dir} 폴더를 찾지 못함.")

    # .pth 파일 모으기
    pth_files = list(target_dir.glob("*.pth"))
    pth_files = [f for f in pth_files if f.resolve() != Path(output_path).resolve()]

    if not pth_files:
        sys.exit("pth 파일이 없음.")

    print(f"{len(pth_files)}개의 가중치파일을 찾음.")

    # 가중치 로드
    collected_weights = []
    for pth in pth_files:
        try:
            collected_weights.append(torch.load(pth, map_location='cpu'))
        except Exception as e:
            print(f"가중치 로드 실패. 파일 이름 : {pth.name}")

    if not collected_weights:
        sys.exit("가중치 파일 없음.")

    # 서버 상태 초기화
    dummy_model = GastroNet(num_classes=3, pretrained=False)
    server = GastroServer(dummy_model, data_root=".", target_organ="unknown")

    # FedAvg 알고리즘 수행
    new_global_weights = server.aggregate_weights(collected_weights)
    
    if new_global_weights is None:
        sys.exit("가중치 통합 실패.")
    
    # 합산한 가중치 모델 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_global_weights, output_path)
    print(f"가중치 저장됨. 위치 : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir", type=str, help=".pth 파일이 저장된 폴더를 넣어주세요.")
    parser.add_argument("--output", type=str, default=None, help="합산된 가중치 파일을 저장할 위치를 알려주세요.")
    
    args = parser.parse_args()

    # Default
    final_output = args.output if args.output else Path(args.weights_dir) / "aggregated_model.pth"
    
    run_aggregation(args.weights_dir, final_output)