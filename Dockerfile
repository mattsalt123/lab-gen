ARG VARIANT=slim-bookworm
ARG VERSION=3.11
FROM python:${VERSION}-${VARIANT} AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.8.23 /uv /uvx /bin/

RUN groupadd -g 10001 genapp && \
    useradd -u 10000 -g genapp genapp
# && chown -R genapp:genapp /app
USER genapp:genapp
WORKDIR /home/genapp

FROM builder
COPY --chown=genapp:genapp . .
RUN uv sync --no-default-groups --frozen
ENV PATH="/home/genapp/.local/bin:${PATH}"
# Start the app
EXPOSE 8000:8000
CMD ["/home/genapp/.venv/bin/python", "-m", "lab_gen"]
