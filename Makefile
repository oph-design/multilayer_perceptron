DIR  = env
NAME = $(DIR)/bin/activate
RSRC = datasets
RSLT = results
CONF = configs/example.conf
DATA = ressources/data.csv

GREEN	= \033[0;32m
WHITE	= \033[0m

all: $(NAME)

$(NAME): $(RSRC) $(RSLT)
	@python3 -m pip install --upgrade pip
	@python3 -m pip install virtualenv
	@virtualenv $(DIR)
	@$(DIR)/bin/python3 -m pip install -r requirements.txt
	@echo "$(GREEN)Dependencies successfully installed!$(WHITE)"

$(RSRC):
	@mkdir -p $(RSRC)

$(RSLT):
	@mkdir -p $(RSLT)
	@mkdir -p $(RSLT)/models/
	@mkdir -p $(RSLT)/evals/


run: $(NAME)
	@. $(NAME); python3 src

clean:
	@find ./src -type d -name "__pycache__" -exec rm -rf {} +
	@echo "$(GREEN)Caches cleaned!$(WHITE)"

fclean: clean
	@rm -rf $(DIR)
	@rm -rf datasets
	@echo "$(GREEN)Virtual environment cleaned!$(WHITE)"

re: fclean all

.PHONY: all clean fclean run test re

