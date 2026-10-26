PY := .venv/bin/python
UV := uv pip install --python $(PY)

.PHONY: install model ui test serve dev build scan

install:
	uv venv --python 3.11 .venv
	$(UV) -e "backend[dev]"
	$(UV) -e "backend[faces]"
	cd frontend && npm install

model:
	$(PY) -m culler.cli fetch-model

ui:
	cd frontend && npm run build

test:
	cd backend && ../$(PY) -m pytest -q

serve:
	$(PY) -m culler.cli serve --port 8000

dev:
	cd frontend && npm run dev

build: ui test

scan:
	$(PY) -m culler.cli scan $(FOLDER) --preset $(or $(PRESET),wedding)
