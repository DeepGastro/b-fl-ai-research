import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import GastroDataset
from src.models.model import GastroNet
from tqdm import tqdm
import os

class GastroClient:
    # clinet_id : (ex: 'hospital_a')
    # device (torch.device): MPS 또는 CPU
    def __init__(self, client_id, data_dir, device=None):
        self.client_id = client_id
        self.data_dir = data_dir
        
        # 해당 프로젝트에선 MPS 가속 사용
        if device:
            self.device = device
        else:
            # CUDA
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("CUDA로 학습 진행")
            # MPS
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("MPS로 학습 진행")
            # CPU
            else:
                self.device = torch.device("cpu")
                print("CPU로 학습 진행")

        # 데이터 로드
        self.train_dataset = GastroDataset(root_dir=data_dir, mode='train')
        self.val_dataset = GastroDataset(root_dir=data_dir, mode='val')
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        
        # 모델 초기화
        self.model = GastroNet(num_classes=3, pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
    
    # ==================서버====================
    
    # Client 가중치 보내기
    def get_weights(self):
        # CPU로 옮겨서 보냄
        return self.model.cpu().state_dict()

    # 메인 서버 가중치 받기
    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.to(self.device) # 다시 내 device(ex. MPS)로 가져오기
        print("서버로부터 가중치 받아옴.")
        
    # ==================서버====================

    def train(self, epochs=1, lr=0.001):
        self.model.train()
        self.model.to(self.device) # 학습 전 device로 이동 확인
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"[{self.client_id}] 병원 학습 시작")
    
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 진행률 바 표시
            progress_bar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix(loss=loss.item())

            epoch_acc = 100 * correct / total
            print('='*50)
            print(f"[{self.client_id}] 학습 결과: Loss {running_loss/len(self.train_loader):.4f} | Acc {epoch_acc:.2f}%")

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = loss / len(self.val_loader)
        print(f"[{self.client_id}] 평가 결과: Loss {avg_loss:.4f} | Acc {acc:.2f}%")
        return acc, avg_loss

if __name__ == "__main__":
    from pathlib import Path
    
    # 경로 테스트
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    test_data_dir = project_root / "data" / "raw" / "stomach" / "federated" / "hospital_a" / "images"
    
    if test_data_dir.exists():
        # 병원 생성
        client = GastroClient("Test_Hospital", str(test_data_dir))
        
        # 학습
        client.train(epochs=1)
        
        # 가중치 추출
        my_weights = client.get_weights()
        print(f"추출된 가중치 타입: {type(my_weights)}")
        
        # 평가
        client.evaluate()
    else:
        print("데이터 경로를 찾지 못함.")