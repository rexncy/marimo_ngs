#!/bin/sh
docker compose -f docker-compose.prod.yml stop && gh repo sync && docker compose -f docker-compose.prod.yml build && docker compose -f docker-compose.prod.yml up