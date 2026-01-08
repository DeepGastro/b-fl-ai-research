import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GastroDataset(Dataset):
    # class_map (dict, optional): {'궤양':0, '암':1, '용종':2} 매핑 정보
    def __init__(self, root_dir, mode='train', class_map=None):
       
        self.root_dir = root_dir # 이미지 폴더 경로
        self.mode = mode # train 또는 val
        self.image_paths = []
        self.labels = []
        
        # 클래스(질병) 이름 찾기 
        # ex) classes = ['궤양', '암', '용종'] (가나다순 정렬)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 클래스를 숫자로 매핑 (궤양->0, 암->1, 용종->2)
        # 연합학습 시 모든 병원이 통일된 번호를 써야 하므로 class_map을 외부에서 받을 수도 있게 함
        if class_map:
            self.class_to_idx = class_map
        else:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 모든 이미지 파일 경로 읽어오기
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if cls_name not in self.class_to_idx:
                continue
                
            cls_idx = self.class_to_idx[cls_name]
            
            # 폴더 내 이미지 파일 탐색
            files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for f in files:
                self.image_paths.append(os.path.join(cls_folder, f))
                self.labels.append(cls_idx)

        print(f"[{mode}] 데이터셋 로드 완료. 경로: {root_dir}")
        print(f"클래스 맵핑: {self.class_to_idx}")
        print(f"총 {len(self.image_paths)}장의 이미지 발견")

        # 이미지 변환(Transform) 정의(EfficientNet)
        # Training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                
                # 상하좌우 반전
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.RandomVerticalFlip(p=0.5),   
                
                # 회전
                transforms.RandomRotation(30),     
                
                # 색감 변화
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                
                transforms.ToTensor(),             
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        # Validation
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 실제 데이터를 달라고 할 때 실행되는 함수
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 이미지 열기 (RGB 모드)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 이미지가 깨져있을 경우에도 멈추지 않고 학습
            print(f"{img_path} 경로 이미지 로드 에러 -> skip함")
            return self.__getitem__(0) 
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
            
        # 텐서와 라벨 반환
        return image, torch.tensor(label, dtype=torch.long)