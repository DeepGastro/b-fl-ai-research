import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from src.models.model import GastroNet
from src.data.dataset import GastroDataset

'''
pretrain.py는 사전 학습용 파일이며 해당 파일로 부터 pre-trained된 가중치를 얻습니다.
설정한 경로에 가중치 파일을 저장하며 이를 맨 처음 메인 서버에서 각 clinet에 보냅니다.
'''

def pretraining(target_organ, epochs=20, batch_size=32):
    # 디바이스 정의
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\n[Pre-train] 목표 장기: {target_organ.upper()} (Device: {device})")
    
    # 경로 설정
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    
    # Train -> Central (90%)
    train_dir = project_root / "data" / "raw" / target_organ / "central" / "images"
    # Validation
    val_dir = project_root / "data" / "raw" / target_organ / "validation" / "images"
    
    # 모델 저장 경로
    save_dir = project_root / "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f"pretrained_{target_organ}.pth"

    # 데이터셋 로드
    if not train_dir.exists():
        print(f"Train 경로 없음: {train_dir}")
        return
    
    train_dataset = GastroDataset(root_dir=str(train_dir), mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_dir.exists():
        val_dataset = GastroDataset(root_dir=str(val_dir), mode='val')
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"데이터 로드: Train({len(train_dataset)}) / Val({len(val_dataset)})")
    else:
        print(f"데이터 로드: Train({len(train_dataset)}) / Val(없음)")

    if len(train_dataset) == 0:
        print("경로를 다시 확인해주세요.")
        return

    # 모델 준비
    model = GastroNet(num_classes=3, pretrained=True).to(device)

    # 클래스 불균형 해결 (Class Weight)
    # [궤양(0), 암(1), 용종(2)] -> 암 데이터가 2배 많으므로 궤양과 용종 가중치를 2배로 높임
    class_weights = torch.tensor([2.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1
    )

    # Training Loop
    print(f"사전 학습 시작 (총 {epochs} Epochs)")
    best_loss = float('inf') 

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        avg_val_loss = 0.0
        val_acc = 0.0
        
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_running_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            print(f"Result: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Acc {val_acc:.2f}%")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                print(f"Best Model 갱신 ({best_loss:.4f} -> {avg_val_loss:.4f}).")
                best_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Result: Train Loss {avg_train_loss:.4f}")
            torch.save(model.state_dict(), save_path)

    print(f"\n최종 학습 종료. 저장된 모델: {save_path}")

if __name__ == "__main__":
    # pretraining(target_organ="stomach", epochs=20)
    pretraining(target_organ="colon", epochs=20)