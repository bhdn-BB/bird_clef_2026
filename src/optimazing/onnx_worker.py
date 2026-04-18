import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.models.model_baseline import BirdSoundClassifier


class OnnxOptimizer:
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        backbone_name: str,
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.device = device

        self.model = self._load_model()

    def _load_model(self):
        model = BirdSoundClassifier(
            num_classes=self.num_classes,
            backbone_name=self.backbone_name,
            lr=1e-3, # It is not important
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _build_input_shape(
        self,
        sr: int,
        n_mels: int,
        hop_length: int,
        duration: float,
    ):
        time_steps = int((sr * duration) / hop_length) + 1
        return (1, 1, n_mels, time_steps)

    def export_to_onnx(
        self,
        onnx_path: str,
        sr: int,
        n_mels: int,
        hop_length: int,
        duration: float,
    ):
        input_shape = self._build_input_shape(sr, n_mels, hop_length, duration)

        dummy_input = torch.randn(*input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {
                    0: "batch_size",
                    3: "time_steps",
                },
                "output": {0: "batch_size"},
            },
        )

        print(f"[ONNX] exported: {onnx_path} | shape={input_shape}")

    def quantize_int8(
        self,
        onnx_path: str,
        quantized_path: str,
    ):
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
        )

        print(f"[ONNX] INT8 saved: {quantized_path}")

    def run(
        self,
        onnx_path: str,
        quantized_path: str,
        sr: int,
        n_mels: int,
        hop_length: int,
        duration: float,
    ):
        self.export_to_onnx(
            onnx_path=onnx_path,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            duration=duration,
        )

        self.quantize_int8(onnx_path, quantized_path)