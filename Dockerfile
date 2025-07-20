# Dockerfile
FROM prefecthq/prefect:2-python3.10

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y gcc git libpq-dev && apt-get clean

# Copia y instala dependencias con Poetry (ajusta si usas pip)
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

# Copia el código fuente
COPY . /app
WORKDIR /app

# Ejecutar bash por defecto (o ajusta en docker-compose)
CMD ["bash"]