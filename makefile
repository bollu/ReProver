LATEST_LIGHTING_PATH :=  $(ls -lt lightning_logs/ | head -n 1)

fit_fstar:
	CONTAINER=native python3 -m retrievalfstar.main  fit --config ./retrievalfstar/confs/cli_random.yaml 

predict_fstar_last:
	CONTAINER=native python3 -m retrievalfstar.main predict  \
		--config lightning_logs/version_44/config.yaml \
		--ckpt_path lightning_logs/version_44/checkpoints/last.ckpt/ 
	
# evaluate_fstar:
# 	@echo "running evaluate on checkpoint $(LATEST_LIGHTNING_PATH)"
# 	CONTAINER=native python3 -m retrievalfstar.evaluate \
# 		  --config lightning_logs/$(LATEST_LIGHTNING_PATH)/config.yaml \
# 		  --preds-file  lightning_logs/$(LATEST_LIGHTNING_PATH)/predictions.pickle

# Reprover is split into a predict and evaluate phase, unlike us.
predict_reprover:
	CONTAINER=native python3 \
		  -m retrieva.main predict \
		  --config retrieval/confs/cli_random.yaml \
		  --ckpt_path PATH_TO_RETRIEVER_CHECKPOINT

evaluate_reprover:
	CONTAINER=native python3 -m retrieval.evaluate \
		  --data-path data/leandojo_benchmark/random \
		  --preds-file PATH_TO_PREDICTIONS_PICKLE



