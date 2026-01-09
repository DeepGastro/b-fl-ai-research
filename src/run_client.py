import torch
import argparse
import shutil
import json  
import datetime 
from pathlib import Path
from src.client import GastroClient

'''
run_client.py파일은 각 client에서 훈련을 할 때 사용합니다.
main_weights 경로에서 서버로 부터 받은 가중치를 불러옵니다.
만약 맨 처음 라운드(pre-trained된 가중치)인 경우에는 해당 가중치를 가져옵니다.
이후 각 client에서 각자의 데이터로 학습을 진행합니다.
학습한 이후에는 가중치와 메타데이터 반환하고 이를 해당 경로에 저장합니다.

실행 예시
python -m src.run_client --id hospital_a --organ colon
'''

def run_client_training(client_id, target_organ, epochs=1):
    # 경로 설정
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    data_root = project_root / "data" / "raw"
    
    # 저장 경로
    base_save_dir = project_root / "saved_models" / target_organ 
    main_weights_dir = base_save_dir / "main_weights"
    client_weights_dir = base_save_dir / "client_weights"
    
    main_weights_dir.mkdir(parents=True, exist_ok=True)
    client_weights_dir.mkdir(parents=True, exist_ok=True)

    # Main 모델 없으면 Pretrained 복사
    global_model_path = main_weights_dir / "global_model_broadcast.pth"
    
    if not global_model_path.exists():
        print(f"받아온 가중치가 없습니다.")
        pretrained_path = project_root / "saved_models" / f"pretrained_{target_organ}.pth"
        
        if pretrained_path.exists():
            print(f"{pretrained_path.name}(pre-trained) 모델을 사용합니다. ")
            shutil.copy(pretrained_path, global_model_path)
        else:
            print(f"{pretrained_path.name}(pre-trained) 모델도 없습니다.")
            return

    print(f"[병원: {client_id}] 학습 시작 (Target: {target_organ.upper()})")
    
    # 학습 진행
    data_dir = data_root / target_organ / "federated" / client_id / "images"
    client = GastroClient(client_id=client_id, data_dir=str(data_dir))
    
    global_weights = torch.load(global_model_path)
    client.set_weights(global_weights)
    
    # train 함수가 결과를 반환하지 않을 경우를 대비해 try-except 처리
    try:
        # client.py의 train함수가 history를 리턴하면 그걸 쓰고, 아니면 그냥 실행
        history = client.train(epochs=epochs)
        # 마지막 epoch의 정확도를 가져오려고 시도 (history가 딕셔너리라면)
        final_acc = history.get('acc', 0.0) if isinstance(history, dict) else 0.0
    except:
        # 리턴값이 없는 구형 코드라면 그냥 실행만 함
        client.train(epochs=epochs)
        final_acc = "N/A" # 정확도 기록 불가
    
    # 결과(가중치) 저장 
    save_path = client_weights_dir / f"{client_id}_update.pth"
    client_weights = client.get_weights()
    torch.save(client_weights, save_path)
    
    # 메타데이터 저장
    json_path = client_weights_dir / f"{client_id}_update.json"
    metadata = {
        "hospital_id": client_id,
        "organ": target_organ,
        "round_accuracy": final_acc,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": save_path.name
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"[병원: {client_id}] 결과 제출 완료")
    print(f"가중치: {save_path.name}")
    print(f"메타데이터: {json_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--organ", type=str, default="colon")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    run_client_training(args.id, args.organ, args.epochs)