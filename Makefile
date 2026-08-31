.PHONY: book-check pdf pdf-check check clean

PYTHON ?= python3

book-check:
	$(PYTHON) scripts/check_book.py

pdf:
	$(PYTHON) scripts/build_pdf.py

pdf-check: pdf
	$(PYTHON) scripts/check_pdf.py

check: book-check pdf-check

clean:
	rm -rf build
	rm -f dist/robot-rl-self-study.pdf
	rm -f dist/SHA256SUMS
