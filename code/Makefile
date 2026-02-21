REMOTE := office
REMOTE_DIR := ~/JPML/code
LOCAL_DIR := /Users/haowuduan/Documents/githubrepos/JPMLCOE/code

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
	rsync -avz --progress $(REMOTE):$(REMOTE_DIR)/outputs/dpf/ $(LOCAL_DIR)/outputs/dpf/

remote-setup:
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'

.PHONY: push push-dry pull-results remote-setup
