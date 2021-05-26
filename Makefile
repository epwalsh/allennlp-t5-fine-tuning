CONFIG = config.jsonnet
SER = ser/
DOCKER_IMAGE_NAME = epwalsh/allennlp-t5:latest
DOCKER_GPUS = --gpus all

.PHONY : check
check :
	@echo 'Checking "$(CONFIG)"'
	@python -c 'from allennlp.common.params import Params; Params.from_file("$(CONFIG)")'

.PHONY : train
train :
	@rm -rf $(SER)
	@allennlp train $(CONFIG) -s $(SER)

.PHONY : docker-build
docker-build :
	docker build -f Dockerfile -t $(DOCKER_IMAGE_NAME) .

.PHONY : docker-push
docker-push :
	docker push $(DOCKER_IMAGE_NAME)

.PHONY : docker-train
docker-train :
	docker run $(DOCKER_GPUS) $(DOCKER_IMAGE_NAME)
