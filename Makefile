NAME = env

GREEN		= \033[0;32m
WHITE		= \033[0m

all: $(NAME)

$(NAME):
	@python3 -m pip install --upgrade pip
	@python3 -m pip install virtualenv
	@virtualenv $(NAME)
	@$(NAME)/bin/python3 -m pip install -r requirements.txt
	@echo "$(GREEN)dependencies successfully installed$(WHITE)"

clean: 
	@rm -rf $(NAME)
	@echo "$(GREEN)Virtual environment cleaned!$(WHITE)"

re: clean all

.PHONY: all clean re
