stg1_ckpt=/home/shiqilin/llm/qwen-ocr/LLaMA-Factory/saves/qwen2.5_vl-3b-ocr-stg1/lora/sft/checkpoint-5500
stg2_ckpt=/home/shiqilin/llm/qwen-ocr/LLaMA-Factory/saves/qwen2.5_vl-3b-ocr-stg2/lora/sft/checkpoint-6000

python eval.py  --inference-results baseline_hmer_inference_results.json --load-hme
python eval.py  --inference-results baseline_unimer_inference_results.json --load-unimer
python eval.py  --inference-results baseline_mer_1k_inference_results.json
python eval.py  --inference-results lora_v8_5500_hmer_inference_results.json --eval-lora --load-hme --lora-ckpt $stg1_ckpt
python eval.py  --inference-results lora_v8_5500_unimer_inference_results.json --eval-lora --load-unimer --lora-ckpt $stg1_ckpt
python eval.py  --inference-results lora_v8_5500_mer_1k_inference_results.json --eval-lora --lora-ckpt $stg1_ckpt
python eval.py  --inference-results lora_v8_cont_6000_hmer_inference_results.json --eval-lora --load-hme --lora-ckpt $stg2_ckpt
python eval.py  --inference-results lora_v8_cont_6000_unimer_inference_results.json --eval-lora --load-unimer --lora-ckpt $stg2_ckpt
python eval.py  --inference-results lora_v8_cont_6000_mer_1k_inference_results.json --eval-lora --lora-ckpt $stg2_ckpt