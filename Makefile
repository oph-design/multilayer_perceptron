NAME = env

GREEN		= \033[0;32m
WHITE		= \033[0m

all: $(NAME)

$(NAME):
	@mkdir -p datasets
	@python3 -m pip install --upgrade pip
	@python3 -m pip install virtualenv
	@virtualenv $(NAME)
	@$(NAME)/bin/python3 -m pip install -r requirements.txt
	@echo "$(GREEN)Dependencies successfully installed$(WHITE)"

run: $(NAME)
	@. $(NAME)/bin/activate; python3 src

clean: 
	@rm -rf datasets
	@rm -rf $(NAME)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "$(GREEN)Virtual environment cleaned!$(WHITE)"

re: clean all

.PHONY: all clean re
