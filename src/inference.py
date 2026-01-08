import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.models.model import GastroNet

class GastroPredictor:
    def __init__(self, target_organ):
        self.target_organ = target_organ
        # CPU/GPU 자동 설정 (백엔드 서버 환경에 맞춤)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[{target_organ.upper()}] NVIDIA GPU (CUDA) 감지됨.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"[{target_organ.upper()}] Apple Silicon (MPS) 감지됨.")
        else:
            self.device = torch.device("cpu")
            print(f"GPU 감지 안됨. [{target_organ.upper()}] CPU 사용.")
        
        # 메인 서버로부터 가져온 가중치 경로
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent
        model_path = project_root / "saved_models" / target_organ / "main_weights" / "global_model_broadcast.pth"
        
        # 모델 구조 생성 및 가중치 로드
        self.model = GastroNet(num_classes=3, pretrained=False)
        
        if model_path.exists():
            weights = torch.load(model_path, map_location=self.device) # 가중치 적용
            self.model.load_state_dict(weights)
            print(f"AI 모듈 로드 완료: {target_organ.upper()}")
        else:
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 이미지 전처리 (AI 파트에서 정한 규칙)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 결과 텍스트 매핑
        self.labels = {0: '궤양', 1: '암', 2: '용종'}

    def predict_image(self, image_path):
        # 이미지 가져오기
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB') # 이미 PIL 객체인 경우
        # Img to Tensor
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob, class_idx = torch.max(probabilities, 1)
            
        # 결과 반환
        result_text = self.labels[class_idx.item()]
        confidence = prob.item() * 100
        
        # AI가 백엔드에게 주는 최종 문자열
        return f"{result_text} (확률: {confidence:.2f}%)"