import torch
import torch.nn as nn
from torchvision import models

class GastroNet(nn.Module):
    # pretrained : 사전 학습 가중치 로드 여부
    def __init__(self, num_classes=3, pretrained=True):
        super(GastroNet, self).__init__()
        
        # EfficientNet-B0 모델 불러오기
        if pretrained:
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            print(f"EfficientNet-B0 사전 학습 가중치 로드됨.")
        else:
            print(f"가중치 초기화 상태로 시작")
            self.model = models.efficientnet_b0(weights=None)

        # 모델의 마지막 층(Classifier) 수정
        in_features = self.model.classifier[1].in_features # B0에선 1280
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), # 과적합 방지
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x) # 배치 이미지에 대해 점수(Logits)반환

if __name__ == "__main__":
    # 테스트 코드
    net = GastroNet(num_classes=3)
    dummy_data = torch.randn(2, 3, 224, 224) # 가짜 이미지 2장 (배치크기 2, 채널 3, 224x224)
    output = net(dummy_data)
    
    print("모델 테스트 완료")
    print(f"입력 크기: {dummy_data.shape}")
    print(f"출력 크기: {output.shape} -> (배치크기, 클래스개수)") # 예상 출력: torch.Size([2, 3])