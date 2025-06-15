build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f api worker beat tusd

test:
	pytest --maxfail=1 --disable-warnings -q
