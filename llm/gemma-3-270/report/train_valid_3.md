환경 : train data set : 1000개, epoch : 20
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
{'loss': 2.9313, 'grad_norm': 2.8012919425964355, 'learning_rate': 0.000182, 'entropy': 2.0066384464502334, 'num_tokens': 11072.0, 'mean_token_accuracy': 0.5644950941205025, 'epoch': 2.0}
{'loss': 1.4893, 'grad_norm': 3.016127109527588, 'learning_rate': 0.000162, 'entropy': 1.6224524825811386, 'num_tokens': 22144.0, 'mean_token_accuracy': 0.7392490342259407, 'epoch': 4.0}
{'loss': 0.8306, 'grad_norm': 2.049722194671631, 'learning_rate': 0.000142, 'entropy': 0.8266381397843361, 'num_tokens': 33216.0, 'mean_token_accuracy': 0.8619220927357674, 'epoch': 6.0}
{'loss': 0.6648, 'grad_norm': 1.0858858823776245, 'learning_rate': 0.000122, 'entropy': 0.5918217703700066, 'num_tokens': 44288.0, 'mean_token_accuracy': 0.8850051447749138, 'epoch': 8.0}
{'loss': 0.5941, 'grad_norm': 1.142496109008789, 'learning_rate': 0.00010200000000000001, 'entropy': 0.570139329880476, 'num_tokens': 55360.0, 'mean_token_accuracy': 0.8914904341101646, 'epoch': 10.0}
{'loss': 0.541, 'grad_norm': 1.197383165359497, 'learning_rate': 8.2e-05, 'entropy': 0.5478507339954376, 'num_tokens': 66432.0, 'mean_token_accuracy': 0.9002802193164825, 'epoch': 12.0}
{'loss': 0.503, 'grad_norm': 1.6906235218048096, 'learning_rate': 6.2e-05, 'entropy': 0.5171513766050339, 'num_tokens': 77504.0, 'mean_token_accuracy': 0.9075192645192146, 'epoch': 14.0}
{'loss': 0.4736, 'grad_norm': 1.2677693367004395, 'learning_rate': 4.2e-05, 'entropy': 0.4969960905611515, 'num_tokens': 88576.0, 'mean_token_accuracy': 0.9111462756991386, 'epoch': 16.0}
{'loss': 0.4504, 'grad_norm': 1.0490247011184692, 'learning_rate': 2.2000000000000003e-05, 'entropy': 0.4782369412481785, 'num_tokens': 99648.0, 'mean_token_accuracy': 0.9159425124526024, 'epoch': 18.0}
{'loss': 0.4398, 'grad_norm': 1.1777116060256958, 'learning_rate': 2.0000000000000003e-06, 'entropy': 0.473800241202116, 'num_tokens': 110720.0, 'mean_token_accuracy': 0.918092705309391, 'epoch': 20.0}
{'train_runtime': 3213.6254, 'train_samples_per_second': 0.249, 'train_steps_per_second': 0.031, 'train_loss': 0.8918049764633179, 'epoch': 20.0}
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [53:33<00:00, 32.14s/it]
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
{"date": "2026-03-04", "time": "19:00", "location": "판교 본사", "attendees": null}

메시지 입력: 내일 오후 3시에 강남역에서 영희랑 커피 마시기로 함
Setting `pad_token_id` to `eos_token_id`:1 for open-end generation.

[추출된 JSON 결과]
{"date": "2026-02-25", "time": "14:00", "location": "강남역", "attendees": ["영희"]}