import onnx
input_file="results/S1465_ProstT5/snapshots/onnx/ProstT5_Finetuning.onnx"
print(onnx.checker.check_model(input_file))

input_file="results/S1465_MSA/snapshots/onnx/MSA_Finetuning.onnx"
print(onnx.checker.check_model(input_file))

input_file="results/S1465_ESM2/snapshots/onnx/ESM2_Finetuning.onnx"
print(onnx.checker.check_model(input_file))
