import json, sys, os, torch, traceback
os.chdir('/home/chinux/jupyterlab/meerkatai')
sys.path.insert(0, '/home/chinux/jupyterlab/meerkatai')

base = 'models/Qwen3.8-27B'
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print('transformers:', __import__('transformers').__version__)
    print('Loading tokenizer...')
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    print('Tokenizer OK')
    print('Loading model with AutoModelForCausalLM...')
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda:0', trust_remote_code=True)
    print('Model loaded:', type(m))
    if hasattr(m, 'language_model'):
        print('Has language_model sub-module')
    if hasattr(m, 'visual'):
        print('Has visual sub-module')
    if hasattr(m, 'mtp'):
        print('Has mtp sub-module')
except Exception as e:
    traceback.print_exc()
    print('ERROR:', e)
