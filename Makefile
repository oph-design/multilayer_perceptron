NAME = env
RSRC = datasets
CONF = configs/example.conf
DATA = ressources/data.csv

GREEN	= \033[0;32m
WHITE	= \033[0m

all: $(NAME)

$(NAME): $(RSRC)
	@python3 -m pip install --upgrade pip
	@python3 -m pip install virtualenv
	@virtualenv $(NAME)
	@$(NAME)/bin/python3 -m pip install -r requirements.txt
	@echo "$(GREEN)Dependencies successfully installed$(WHITE)"

$(RSRC):
	@mkdir -p $(RSRC)

run: $(NAME)
	@. $(NAME)/bin/activate; python3 src

test: $(NAME)
	@. $(NAME)/bin/activate; python3 src $(CONF) $(DATA) 0.8

clean:
	@rm -rf datasets
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "$(GREEN)Caches cleaned!$(WHITE)"

fclean: clean
	@rm -rf $(NAME)
	@echo "$(GREEN)Virtual environment cleaned!$(WHITE)"

re: fclean all

.PHONY: all clean fclean run test re
