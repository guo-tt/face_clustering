PROJECT=face
REPO=$(PROJECT)
TAG=0.0.8

.PHONY: deploy build

test:
	docker run -it --rm \
	--name $(PROJECT)-test \
	-p 8899:8888 \
	-v `pwd`:/srv \
	$(REPO):$(TAG) \
    bash
	
build:
	docker build -f Dockerfile -t $(REPO):$(TAG) .

clean:
	find . \( -name \*.pyc -o -name \*.pyo -o -name __pycache__ \) -prune -exec rm -rf {} +