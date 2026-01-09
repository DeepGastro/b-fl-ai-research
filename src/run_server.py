import torch
from pathlib import Path
from src.models.model import GastroNet
from src.server import GastroServer
import datetime

'''
해당 파일은 메인 서버에서 돌리는 코드입니다.
실행시 받은 가중치들은 한 번에 모아서 평균을 내고 해당 가중치를 저장(덮어쓰움)합니다.
나온 가중치를 각 client에 보내면 됩니다.
archive에 백업도 할 수 있는 기능도 있습니다.
'''

def run_aggregation(target_organ):
    # 경로 설정
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    data_root = project_root / "data" / "raw"
    
    # 장기(target_organ)별로 폴더 분리
    base_save_dir = project_root / "saved_models" / target_organ
    client_weights_dir = base_save_dir / "client_weights"
    main_weights_dir = base_save_dir / "main_weights"
    
    main_weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Server] {target_organ.upper()} 통합(Aggregation) 시작")

    # 모델 준비
    global_model = GastroNet(num_classes=3, pretrained=True)
    server = GastroServer(global_model, [], str(data_root), target_organ)
    
    # 파일 수집
    collected_weights = []
    hospital_list = ["hospital_a", "hospital_b", "hospital_c"]
    
    missing_files = False
    for h_id in hospital_list:
        file_path = client_weights_dir / f"{h_id}_update.pth"
        
        if file_path.exists():
            weights = torch.load(file_path)
            collected_weights.append(weights)
            print(f"수신: {h_id}_update.pth")
        else:
            print(f"미수신: {h_id} (파일 없음)")
            missing_files = True

    if missing_files:
        print("파일 누락으로 인해 통합 중단됨.")
        return

    # FedAvg 알고리즘 수행
    new_global_weights = server.aggregate_weights(collected_weights)
    server.global_model.load_state_dict(new_global_weights)
    print("FedAvg 적용됨.")
    
    # 합산한 가중치 모델 저장
    new_global_path = main_weights_dir / "global_model_broadcast.pth"
    torch.save(new_global_weights, new_global_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    archive_dir = base_save_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = archive_dir / f"round_model_{timestamp}.pth"
    torch.save(new_global_weights, archive_path)
    
    print(f"가중치 통합 완료")
    print(f"배포용 업데이트: {new_global_path.name}")
    print(f"기록용 백업완료: {archive_path.name}")

if __name__ == "__main__":
    TARGET_ORGAN = "colon" 
    # TARGET_ORGAN = "stomach"
    
    run_aggregation(TARGET_ORGAN)