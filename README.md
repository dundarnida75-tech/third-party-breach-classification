# Third-Party Breach News Classification

Binary text classification using fine-tuned Small Language Models.

## Models Evaluated
- SmolLM2-360M
- TinyLlama-1.1B  
- Qwen2.5-1.5B
- Gemma 4 E2B

## Usage
1. `python data_prep.py` — prepare dataset
2. `python train.py --model smollm2` — fine-tune
3. `python evaluate.py --compare outputs/smollm2 outputs/tinyllama` — evaluate

## Results
*(tablo buraya gelecek — sonuçlar çıkınca ekle)*
