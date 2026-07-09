환경 : train data set : 240개, epoch : 10
목적 : JSON 형태 대답
개선 방향 : 데이터 셋 증가, epochs 수 증가

(venv) C:\WorkSpace\Dev\Python\AIStudy>python llm\gemma-3-270\train.py
🚀 [C:\WorkSpace\Dev\Python\AIStudy\models\gemma-3-270m-it] 토크나이저 로드 중...
💻 CPU에 모델 로드 중 (메모리 약 1GB 필요)...
`torch_dtype` is deprecated! Use `dtype` instead!
📂 데이터셋 [C:\WorkSpace\Dev\Python\AIStudy\llm\gemma-3-270\schedule_dataset.jsonl] 로드 중...
The model is already on multiple devices. Skipping the move to device specified in `args`.
🏃 CPU 환경에서 파인튜닝을 시작합니다...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 2, 'pad_token_id': 0}.
{'loss': 3.681, 'grad_norm': 2.8327653408050537, 'learning_rate': 0.000164, 'entropy': 1.5363457590341567, 'num_tokens': 11072.0, 'mean_token_accuracy': 0.5155870303511619, 'epoch': 2.0}
{'loss': 2.727, 'grad_norm': 2.4389808177948, 'learning_rate': 0.000124, 'entropy': 2.1676846921443937, 'num_tokens': 22144.0, 'mean_token_accuracy': 0.5544308751821518, 'epoch': 4.0}
{'loss': 2.2385, 'grad_norm': 1.8888204097747803, 'learning_rate': 8.4e-05, 'entropy': 2.209692034125328, 'num_tokens': 33216.0, 'mean_token_accuracy': 0.6224169984459877, 'epoch': 6.0}
{'loss': 1.9237, 'grad_norm': 1.9550470113754272, 'learning_rate': 4.4000000000000006e-05, 'entropy': 1.946099618077278, 'num_tokens': 44288.0, 'mean_token_accuracy': 0.676312755048275, 'epoch': 8.0}
{'loss': 1.7516, 'grad_norm': 1.9750195741653442, 'learning_rate': 4.000000000000001e-06, 'entropy': 1.8237234801054, 'num_tokens': 55360.0, 'mean_token_accuracy': 0.6936842232942582, 'epoch': 10.0}
{'train_runtime': 1648.8905, 'train_samples_per_second': 0.243, 'train_steps_per_second': 0.03, 'train_loss': 2.4643506240844726, 'epoch': 10.0}
100%|████████████████████████████████████████████████████████████████████████████████| 50/50 [27:28<00:00, 32.98s/it]
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
null


메시지 입력: 내일 오후 3시에 강남역에서 영희랑 커피 마시기로 함
Setting `pad_token_id` to `eos_token_id`:1 for open-end generation.

[추출된 JSON 결과]
null