import sys
sys.path.insert(0, 'D:/Ming-3.0-tts')
try:
    import onnxruntime
    print("onnxruntime", onnxruntime.__version__, onnxruntime.__file__)
except Exception as e:
    print("onnxruntime FAIL", e)
try:
    import torchaudio.compliance.kaldi as kaldi
    print("kaldi ok", kaldi.__file__)
except Exception as e:
    print("kaldi FAIL", e)
