# ----------------------------------
#          INSTALL
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* kaggle_house_prices_competition/*.py

black:
	@black scripts/* kaggle_house_prices_competition/*.py

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr kaggle_house_prices_competition-*.dist-info
	@rm -fr kaggle_house_prices_competition.egg-info

install:
	@pip install . -U

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      KAGGLE
# ----------------------------------

COMPETITION_REF=house-prices-advanced-regression-techniques

list_files:
	@kaggle competitions files ${COMPETITION_REF}

download_files:
	@kaggle competitions download -p raw_data/ ${COMPETITION_REF}
	@unzip "raw_data/*.zip" -d raw_data
	@find raw_data/ -name "*.zip" -type f -delete
