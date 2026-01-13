import torch
import copy
from pathlib import Path

class GastroServer:
    def __init__(self, global_model, data_root, target_organ):
        self.global_model = global_model
        
        # run_server.py에서 data_root를 "."(dummy)로 넘길 때를 대비해 에러 방지 처리
        try:
            save_dir = Path(data_root).parent.parent / "saved_models"
            model_name = f"pretrained_{target_organ}.pth" 
            model_path = save_dir / model_name
            
            # 파일이 진짜 있을 때만 로드
            if model_path.exists():
                # 안전하게 CPU로 로드
                loaded_weights = torch.load(model_path, map_location='cpu')
                self.global_model.load_state_dict(loaded_weights)
            else:
                pass 

        except Exception:
            pass

    def aggregate_weights(self, weights_list):
        """
        FedAvg 알고리즘: 수집된 가중치 리스트(weights_list)를 평균 냅니다.
        """
        if not weights_list:
            return None
        
        # 첫 번째 가중치를 기준(Base)으로 복사
        avg_weights = copy.deepcopy(weights_list[0])
        
        # 나머지 가중치들을 모두 더함
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            
            # 전체 개수로 나누어 평균 계산
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
            
        return avg_weights