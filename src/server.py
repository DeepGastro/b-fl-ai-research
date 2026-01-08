import torch
import copy
from pathlib import Path
from src.models.model import GastroNet
from src.client import GastroClient

class GastroServer:
    def __init__(self, global_model, client_ids, data_root, target_organ):
        self.global_model = global_model
        self.client_ids = client_ids
        
        # 가중치 파일 불러오기 : pretrained_stomach.pth 또는 pretrained_colon.pth
        save_dir = Path(data_root).parent.parent / "saved_models"
        model_name = f"pretrained_{target_organ}.pth" 
        model_path = save_dir / model_name
        
        print(f"현재 타겟 장기: {target_organ.upper()}")
        
        if model_path.exists():
            loaded_weights = torch.load(model_path)
            self.global_model.load_state_dict(loaded_weights)
            print("가중치 로드됨.")
        else:
            print("로드 실패")
            
        # 클라이언트(병원) 생성
        self.clients = []
        for c_id in client_ids:
            # data/raw/{target_organ}/federated/{hospital}/images
            c_data_dir = f"{data_root}/{target_organ}/federated/{c_id}/images"
            client = GastroClient(client_id=c_id, data_dir=c_data_dir)
            self.clients.append(client)

    def aggregate_weights(self, weights_list):
        # FedAvg 알고리즘
        avg_weights = copy.deepcopy(weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
        return avg_weights

    def train_round(self, round_num, local_epochs=1):
        print(f"\n[연합 학습 진행 : Round {round_num}]")
        
        global_weights = self.global_model.state_dict()
        collected_weights = []
        
        for client in self.clients:
            client.set_weights(global_weights)
            client.train(epochs=local_epochs)
            client.evaluate() 
            collected_weights.append(client.get_weights())
        
        new_global_weights = self.aggregate_weights(collected_weights)
        self.global_model.load_state_dict(new_global_weights)
        print(f"Round {round_num} 완료. 글로벌 모델 업데이트됨")

if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    data_root = project_root / "data" / "raw"
    
    TARGET_ORGAN = "stomach"  # 혹은 "colon"
    # TARGET_ORGAN = "colon"

    print(f"{TARGET_ORGAN} 모델 초기화")
    global_model = GastroNet(num_classes=3, pretrained=True)
    
    server = GastroServer(global_model, ["hospital_a", "hospital_b", "hospital_c"], str(data_root), TARGET_ORGAN)
    
    # 연합 학습 3라운드 진행
    for r in range(1, 4):
        server.train_round(round_num=r, local_epochs=1)
        
    print("\n모든 학습 종료")