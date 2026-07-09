환경 : train data set : 100개, epoch : 3
목적 : JSON 형태 대답
개선 방향 : LORA방식으로 파인튜닝

(venv) C:\WorkSpace\Dev\Python\AIStudy>python llm\gemma-3-270\train.py
🚀 [C:\WorkSpace\Dev\Python\AIStudy\models\gemma-3-270m-it] 토크나이저 로드 중...
💻 CPU에 모델 로드 중 (메모리 약 1GB 필요)...
`torch_dtype` is deprecated! Use `dtype` instead!
📂 데이터셋 [C:\WorkSpace\Dev\Python\AIStudy\llm\gemma-3-270\schedule_dataset.jsonl] 로드 중...
Adding EOS to train dataset: 100%|███████████████████| 40/40 [00:00<00:00, 1289.79 examples/s]
Tokenizing train dataset: 100%|██████████████████████| 40/40 [00:00<00:00, 1966.09 examples/s]
Truncating train dataset: 100%|██████████████████████| 40/40 [00:00<00:00, 8228.16 examples/s] 
The model is already on multiple devices. Skipping the move to device specified in `args`.
🏃 CPU 환경에서 파인튜닝을 시작합니다...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 2, 'pad_token_id': 0}.
{'loss': 3.7669, 'grad_norm': 3.2204062938690186, 'learning_rate': 8e-05, 'entropy': 1.4917951285839082, 'num_tokens': 11072.0, 'mean_token_accuracy': 0.5122352816164494, 'epoch': 2.0}      
{'train_runtime': 661.0751, 'train_samples_per_second': 0.182, 'train_steps_per_second': 0.023, 'train_loss': 3.5780391693115234, 'entropy': 1.757809227705002, 'num_tokens': 16608.0, 'mean_token_accuracy': 0.523830983042717, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████████| 15/15 [11:01<00:00, 44.07s/it] 
✅ 학습 완료! 모델 가중치와 토크나이저를 [C:\WorkSpace\Dev\Python\AIStudy\llm\gemma-3-270\gemma_schedule_extractor\final_lora_weights]에 저장합니다.

🎉 모든 과정이 성공적으로 끝났습니다. 저장 경로: C:\WorkSpace\Dev\Python\AIStudy\llm\gemma-3-270\gemma_schedule_extractor\final_lora_weights

(venv) C:\WorkSpace\Dev\Python\AIStudy>python llm\gemma-3-270\valid_model.py
⏳ 토크나이저와 베이스 모델을 로드합니다...
`torch_dtype` is deprecated! Use `dtype` instead!
🔥 학습된 LoRA 가중치를 모델에 결합합니다...

==================================================
🤖 일정 추출 봇이 준비되었습니다. (종료하려면 'q' 입력)
==================================================

메시지 입력: 모레 저녁에 판교 카카오 본사에서 미팅 있음
Setting `pad_token_id` to `eos_token_id`:1 for open-end generation.

[추출된 JSON 결과]
네, 모레 저녁에 판교 카카오 본사에서 미팅 있음입니다.

메시지 입력: 내일 오후 3시에 강남역에서 영희랑 커피 마시기로 함
Setting `pad_token_id` to `eos_token_id`:1 for open-end generation.

[추출된 JSON 결과]
네, 내일 오후 3시에 강남역에서 영희와 커피를 마시는 것을 추천합니다.