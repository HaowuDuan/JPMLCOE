REMOTE := office
REMOTE_DIR := ~/JPML/code
REMOTE_DIR_JIT := ~/JPML-jit/code
LOCAL_DIR := /Users/haowuduan/Documents/githubrepos/HMC-DPF-OT/code

EXCLUDES := --exclude='.venv/' \
            --exclude='outputs/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='.pytest_cache/' \
            --exclude='.DS_Store'

push:
	rsync -avz --progress $(EXCLUDES) $(LOCAL_DIR)/ $(REMOTE):$(REMOTE_DIR)/

push-dry:
	rsync -avzn --progress $(EXCLUDES) $(LOCAL_DIR)/ $(REMOTE):$(REMOTE_DIR)/

pull-results:
	rsync -avz --progress $(REMOTE):$(REMOTE_DIR)/outputs/ $(LOCAL_DIR)/outputs/

remote-setup:
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'

# JIT branch targets — separate remote directory
push-jit:
	rsync -avz --progress $(EXCLUDES) $(LOCAL_DIR)/ $(REMOTE):$(REMOTE_DIR_JIT)/

push-jit-dry:
	rsync -avzn --progress $(EXCLUDES) $(LOCAL_DIR)/ $(REMOTE):$(REMOTE_DIR_JIT)/

pull-results-jit:
	rsync -avz --progress $(REMOTE):$(REMOTE_DIR_JIT)/outputs/ $(LOCAL_DIR)/outputs/
	mkdir -p $(LOCAL_DIR)/tests/hmc/results
	rsync -avz --progress $(REMOTE):$(REMOTE_DIR_JIT)/tests/hmc/results/ $(LOCAL_DIR)/tests/hmc/results/

remote-setup-jit:
	ssh $(REMOTE) 'mkdir -p $(REMOTE_DIR_JIT) && cd $(REMOTE_DIR_JIT) && python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'

.PHONY: push push-dry pull-results remote-setup push-jit push-jit-dry pull-results-jit remote-setup-jit
