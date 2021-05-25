CONFIG = config.jsonnet
SER = ser/

.PHONY : check
check :
	@echo 'Checking "$(CONFIG)"'
	@python -c 'from allennlp.common.params import Params; Params.from_file("$(CONFIG)")'

.PHONY : train
train :
	@rm -rf $(SER)
	@allennlp train $(CONFIG) -s $(SER)
