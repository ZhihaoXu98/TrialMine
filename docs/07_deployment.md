# Week 7 — Containerized Deployment (Textbook Edition)

> A textbook walkthrough of how we packaged the TrialMine agent system
> into a one-command demo. We'll learn the underlying ideas (containers,
> orchestration, observability) one at a time, with worked examples,
> before showing how each one shows up in our real code.

---

## How to read this textbook

This document has **9 chapters** and **4 appendices**. Each chapter:

- Opens with **learning objectives** (3–4 things you'll know by the end).
- Introduces ideas **one at a time**, with a small example before the
  real code.
- Ends with a **recap** and **check-yourself questions**.

You can read it cover to cover (~90 minutes), or jump to a chapter
when you need it. Each chapter is self-contained.

| Chapter | What you'll learn | Time |
|---|---|---|
| 1. The Goal | What "shipping a demo" means and why it's harder than it looks | 5 min |
| 2. Containers and Images | The mental model behind Docker | 10 min |
| 3. Writing a Dockerfile | How to package one service | 10 min |
| 4. Docker Compose | How to wire many services together | 15 min |
| 5. The Streamlit UI | How the front-end is built | 10 min |
| 6. Observability | Metrics, Prometheus, Grafana | 15 min |
| 7. Cold Starts and Threading | Why the first request is slow, and how to fix it | 10 min |
| 8. The Bug Hunt | Three real bugs and how we found them | 15 min |
| 9. Running the Stack | Step-by-step instructions | 5 min |

**Appendices**:
- A. Map of every file we touched
- B. Five design decisions, plain English
- C. Glossary
- D. Exercises (try these to test understanding)

---

## Chapter 1 — The Goal

> **By the end of this chapter, you'll be able to:**
> 1. Explain what "shipping a demo" requires beyond a working algorithm.
> 2. Name the three pieces we need to add to the existing system.
> 3. Sketch the final architecture from memory.

### 1.1 Where we left off

After Week 6, our agent system worked. You could open a Python REPL,
import `TrialMine.agents.pipeline`, call `search("lung cancer trials")`,
and get back ranked, eligibility-checked results. The smoke test
passed. The unit tests passed.

But here's the awkward truth: **a smoke test passing is not the same
as a system you can show to another human.**

To actually demo this thing, a person needs to:

1. Type in a search.
2. See results in a browser.
3. Trust the system enough to keep using it.

None of that is provided by a Python REPL.

### 1.2 The three things we're missing

Think about every demo you've ever watched. There are three pieces:

```
   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
   │   1. FACE    │       │   2. BOX     │       │   3. EYES    │
   │              │       │              │       │              │
   │ Something    │       │ One command  │       │ When things  │
   │ a person     │       │ brings the   │       │ break, you   │
   │ can type     │       │ whole system │       │ can see why  │
   │ into.        │       │ up.          │       │              │
   └──────────────┘       └──────────────┘       └──────────────┘
        UI                  Deployment            Observability
```

Without all three, you're not demoing — you're explaining. People
forget explanations. They remember demos.

### 1.3 What we're going to build

```
                       ┌──────────────────────────┐
                       │      Streamlit UI        │
                       │  (port 8501, browser)    │
                       └────────────┬─────────────┘
                                    │ HTTP
                       ┌────────────▼─────────────┐
                       │       FastAPI            │
                       │  (port 8000, /search)    │
                       └────┬────────┬────────┬───┘
                            │        │        │
                       ┌────▼─┐  ┌───▼──┐  ┌──▼─────┐
                       │ ES   │  │Redis │  │Anthropic│
                       │ 9200 │  │ 6379 │  │  API    │
                       └──────┘  └──────┘  └─────────┘

                       (And, watching everything:)

                       ┌──────────────────────────┐
                       │ Prometheus → Grafana     │
                       │   9090       3000        │
                       └──────────────────────────┘
```

Six services, all started by **one command**:

```
docker compose up --build
```

That's the goal. The rest of this textbook is how we get there.

### Recap

- A working algorithm and a demoable system are not the same thing.
- We need three things: a UI, single-command deployment, and
  observability.
- Our final stack has six services orchestrated by Docker Compose.

### Check yourself

1. If your boss asks "show me the agent system," is the smoke test
   enough? Why not?
2. Could you draw the architecture diagram from memory?

---

## Chapter 2 — Containers and Images

> **By the end of this chapter, you'll be able to:**
> 1. Explain the difference between a container and an image.
> 2. Describe why we use containers instead of "just running the code".
> 3. Define *layer*, *bind mount*, and *named volume*.

### 2.1 The problem containers solve

Suppose I write a Python program that uses a specific version of
PyTorch, a specific version of an NLP library, and a Linux system
package called `libgomp1`. On my Mac, it works. I send the code to
you. You run it on Ubuntu — different Python version, different
PyTorch, no `libgomp1`. It crashes.

This is the **"works on my machine"** problem.

A **container** is a way to wrap up an entire program *and its
environment* — Python version, libraries, system packages — into a
single unit. Run the container on any computer with Docker installed,
and it runs identically.

> **Mental model**: a container is a tiny, isolated computer running
> just your program.

### 2.2 Image vs container

Two words you'll hear constantly. They mean different things:

```
       IMAGE                              CONTAINER
       ─────                              ─────────

  A frozen template.                A running instance of an image.
  Stored on disk.                   Has memory, processes, state.
  Doesn't run.                      Can be started, stopped, deleted.

  Think: a class.                   Think: an object.
  Think: a recipe.                  Think: a cake.
```

You **build** an image once. You **run** containers from that image
many times.

### 2.3 Layers and caching

An image is made of **layers** stacked on top of each other.

```
  Layer 4 :  Your application code           ← 50 MB
  Layer 3 :  Python packages (pip install)   ← 800 MB
  Layer 2 :  System tools (apt-get install)  ← 100 MB
  Layer 1 :  Base OS (python:3.11-slim)      ← 130 MB
  ────────────────────────────────────────
                                       Total ~1 GB
```

When you change one line in your code, only **Layer 4** is rebuilt.
Layers 1, 2, and 3 are reused from the cache. The next build takes
seconds, not minutes.

This is why the order of steps in a Dockerfile matters — you put
**stable things at the bottom** (rarely change) and **frequently-changing
things at the top** (your code).

### 2.4 Why containers, not just running the code

| Without containers | With containers |
|---|---|
| "Works on my machine" issues | Same image, every machine, identical behaviour |
| Library version conflicts between projects | Each container has its own world |
| Slow VM-style isolation (gigabytes, minutes) | Lightweight (megabytes, milliseconds) |
| Manual deployment scripts | `docker run image-name` |

### 2.5 Volumes: how containers see your data

A container has its own filesystem, but sometimes you want the
container to **read** files from your computer (say, `data/trials.db`)
or **write** files that survive after the container is deleted (say,
the Elasticsearch index).

Two ways to give a container access to disk:

**Bind mount** — point a host directory at a container directory:

```
HOST                              CONTAINER
─────                             ─────────
~/Desktop/TrialMine/data    ──→   /app/data    (read-only, in our case)
```

When the API reads `/app/data/trials.db`, it's actually reading
your `~/Desktop/TrialMine/data/trials.db`. Useful for source code,
data, models — things that already exist on your machine.

**Named volume** — Docker manages the storage:

```
HOST                              CONTAINER
─────                             ─────────
(a directory Docker chose) ←─→   /usr/share/elasticsearch/data
```

You don't know or care where on your machine it is. Docker handles
it. The data **survives** if you delete and recreate the container.
Useful for stateful services (databases, indexes).

### Recap

- An **image** is a frozen template; a **container** is a running
  instance of one.
- Images are built from **layers**; rarely-changing layers go at the
  bottom of a Dockerfile so the cache stays warm.
- **Bind mounts** map host directories into containers; **named
  volumes** are Docker-managed persistent storage.

### Check yourself

1. If you write `RUN pip install torch` *before* `COPY src/ .` in
   your Dockerfile, what happens to the cache when you change a
   line in `src/`?
2. Should `data/trials.db` be in a bind mount or a named volume? Why?
3. If you `docker compose down` and restart, does Elasticsearch
   keep its index? What if you `docker compose down -v`?

---

## Chapter 3 — Writing a Dockerfile

> **By the end of this chapter, you'll be able to:**
> 1. Read and write a basic Dockerfile.
> 2. Explain what `.dockerignore` is for.
> 3. Refactor a single-stage Dockerfile into multi-stage to shrink it.

### 3.1 A naive first Dockerfile

Suppose we want to package our FastAPI service. Naive attempt:

```dockerfile
FROM python:3.11

COPY . /app
WORKDIR /app
RUN pip install .

CMD ["uvicorn", "TrialMine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

This works! But it has problems:

1. **It's huge.** `python:3.11` (without `-slim`) is ~900 MB. Plus
   your code. Plus your dependencies. Plus the `pip` tool itself.
   Plus the entire `apt` cache from when Python was installed.
   Easily 4–5 GB.

2. **It rebuilds slowly.** `COPY . /app` copies *everything* —
   data files, model weights, notebooks, the `.git` folder.
   Changing one line of code invalidates this `COPY` and forces
   `pip install` to re-run.

3. **It runs as root.** Dangerous: a security exploit gets root
   inside the container.

Let's fix these problems one at a time.

### 3.2 Fix 1 — Use a slim base image

```dockerfile
FROM python:3.11-slim                   # ← was python:3.11
```

`python:3.11-slim` strips out the `apt` cache, `man` pages,
documentation, and other things you don't need at runtime. About
130 MB instead of 900 MB.

### 3.3 Fix 2 — `.dockerignore`

Create a file called `.dockerignore` next to your Dockerfile:

```
# .dockerignore
data/
models/
.git/
notebooks/
**/__pycache__
.venv/
docs/
tests/
```

Docker reads this **before** sending files to the build. Anything
matched is **not** copied. The build context goes from 11 GB to a
few hundred KB. Much faster builds.

### 3.4 Fix 3 — Multi-stage build

The Dockerfile we're using to *install* PyTorch and SciSpacy needs
build tools (`gcc`, `build-essential`, `git`). The Dockerfile we
*run* doesn't need those tools — they're only used during install.

**Multi-stage builds** let you separate these. You write multiple
`FROM` blocks, each producing an intermediate image, and you choose
which one becomes the final image.

```dockerfile
# Stage 1: builder. Install everything (heavy).
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y build-essential gcc git

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install .

# Stage 2: runtime. Copy ONLY what we need from the builder.
FROM python:3.11-slim AS runtime

# Just the runtime library (no compiler)
RUN apt-get update && apt-get install -y libgomp1

# Copy the installed packages from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

CMD ["uvicorn", "TrialMine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Diagram of what happens:

```
                  builder stage                   runtime stage
                  ──────────────                   ─────────────
   ┌────────────────────────────────┐    ┌────────────────────────────┐
   │ python:3.11-slim     (130 MB)  │    │ python:3.11-slim   (130 MB)│
   │ + build-essential     (1 GB)   │    │ + libgomp1          (5 MB) │
   │ + gcc, git           (200 MB)  │    │ + Python packages   (3 GB) │
   │ + Python packages     (3 GB)   │ ─→ │   ↑ copied from builder    │
   │ + your code           (1 MB)   │    │                            │
   └────────────────────────────────┘    └────────────────────────────┘
              ~4.3 GB                              ~3.1 GB
              (thrown away)                  (this is your final image)
```

The compiler tooling (1.2 GB) gets thrown away. The runtime stays slim.

### 3.5 Our actual API Dockerfile, annotated

```dockerfile
# syntax=docker/dockerfile:1.7
# ─── Stage 1: builder ───
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1               # don't waste disk on pip's cache

# Install compilers and headers — needed to build faiss, lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only torch in its OWN layer.
# Why its own layer? PyTorch is the heaviest single dependency
# (~200 MB, takes a minute to download). Putting it before our
# project install means: change a line in src/, torch layer stays
# cached, only the Python install layer rebuilds.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Now install our project. Order matters: pyproject.toml first
# (rarely changes), then src/ (changes often).
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install .

# SciSpacy biomedical model — ~600 MB pinned wheel
RUN pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# ─── Stage 2: runtime ───
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1                # default; we override to 8 in compose

# libgomp1 is the only system dep at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Run as a non-root user — defense in depth
RUN useradd --create-home --shell /usr/sbin/nologin trialmine
USER trialmine
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "TrialMine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

A few things to notice:

- We **don't** `COPY src/` into the runtime stage. Why not? Because
  `pip install .` already copied the package into
  `site-packages` — it's installed, not just present. The runtime
  has no source tree, just installed packages.
- We `useradd` and `USER trialmine`. Programs run as `trialmine`,
  not root. If something exploits a bug in the API, the attacker
  has the privileges of `trialmine` (basically nothing).
- `EXPOSE 8000` is documentation — it tells Docker (and humans)
  that this container listens on port 8000. It doesn't actually
  publish the port; that happens in `docker-compose.yml`.

### 3.6 The UI Dockerfile (much simpler)

The UI doesn't need PyTorch, SciSpacy, or any heavy ML library.
Single stage is fine:

```dockerfile
FROM python:3.11-slim

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# wget is for the compose healthcheck
RUN apt-get update && apt-get install -y wget

RUN pip install "streamlit>=1.40,<2" "httpx>=0.27,<1"

RUN useradd --create-home streamlit
USER streamlit
WORKDIR /app

COPY --chown=streamlit:streamlit src/TrialMine/ui/ /app/ui/

EXPOSE 8501
CMD ["streamlit", "run", "/app/ui/app.py"]
```

About 800 MB total.

### Recap

- Use `python:3.11-slim`, not `python:3.11`.
- Use `.dockerignore` to keep multi-GB folders out of the build context.
- Use **multi-stage builds** to leave the build tooling behind.
- Order Dockerfile commands so rarely-changing things come first
  (better cache hits).
- Run as a non-root user.

### Check yourself

1. Why is CPU-only PyTorch installed on its own line, before
   `COPY src/`?
2. If you change `src/TrialMine/api/routes.py`, which Dockerfile
   layers stay cached and which rebuild?
3. The runtime stage doesn't `COPY src/`. How does the FastAPI
   app find its own code?

---

## Chapter 4 — Docker Compose

> **By the end of this chapter, you'll be able to:**
> 1. Explain why we need orchestration when running more than one container.
> 2. Read and write a `docker-compose.yml`.
> 3. Use service hostnames, healthchecks, and volumes correctly.

### 4.1 The problem orchestration solves

So you can run a single container with `docker run`. Great. But our
demo has six services. To start them by hand:

```
docker run elasticsearch ...
docker run redis ...
docker run -e ELASTICSEARCH_URL=... api ...
docker run -e API_URL=... ui ...
docker run -v ./prometheus.yml:... prometheus ...
docker run -v ./grafana/... grafana ...
```

Six terminals. Six different sets of arguments. Easy to forget one.
And what if the API starts before Elasticsearch is ready? It crashes.

**Docker Compose** is one YAML file describing all of this:

```yaml
services:
  elasticsearch: ...
  redis: ...
  api: ...
  ui: ...
  prometheus: ...
  grafana: ...
```

Then:

```
docker compose up
```

That's it. All six services, started in the right order, with the
right config, talking to each other.

### 4.2 The simplest possible compose file

Let's start small. Two services: a web app and a database.

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: secret
  web:
    image: my-app:latest
    environment:
      DATABASE_URL: postgresql://postgres:secret@db:5432/postgres
```

Compose creates a **network** automatically. On that network, every
service is reachable by its **service name** as a DNS hostname.
Notice the `DATABASE_URL`: `db:5432`, not `localhost:5432`. From
inside the `web` container, the hostname `db` resolves to the
postgres container.

This is the most important rule:

> **Inside the compose network, services find each other by service
> name. From outside (your laptop's browser), services are reached
> via `localhost` and the port that compose published.**

### 4.3 Service hostnames vs `localhost`

Here's a table that will save you a debugging session:

| You are... | Trying to reach... | Use this address |
|---|---|---|
| Inside the `api` container | Elasticsearch | `http://elasticsearch:9200` |
| Inside the `api` container | Itself | `http://localhost:8000` |
| Inside the `ui` container | API | `http://api:8000` |
| Your laptop browser | UI | `http://localhost:8501` |
| Your laptop browser | API | `http://localhost:8000` |
| Your laptop browser | Elasticsearch | `http://localhost:9200` |

Hostnames inside the network. Localhost outside.

This is why our code reads `ELASTICSEARCH_URL` from an environment
variable. In dev, you'd set it to `localhost:9200`. In Compose, we
override it to `elasticsearch:9200`. Same code.

```python
# In src/TrialMine/api/app.py
ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
```

### 4.4 Publishing ports

We said the browser accesses things via `localhost:8501`. How does
the host know that the UI container is at `localhost:8501`?

Look at this:

```yaml
ui:
  image: trialmine-ui
  ports:
    - "8501:8501"   #  HOST_PORT : CONTAINER_PORT
```

The `ports` line says: *"forward the host's port 8501 to the
container's port 8501."* Now `localhost:8501` on your laptop reaches
the container.

You can map to different ports:

```yaml
ports:
  - "9001:8501"   # host's 9001 → container's 8501
```

Now you'd visit `localhost:9001`.

### 4.5 Healthchecks — "is the service ready?"

Just because a container has *started* doesn't mean it's *ready*.
Elasticsearch takes 30 seconds after starting before it can serve
queries. If the API container starts faster and sends queries
immediately, they fail.

**Healthchecks** fix this. A healthcheck is a command Docker runs
inside the container periodically. If the command exits 0, the
container is marked **healthy**. If not, it's still **unhealthy**.

```yaml
elasticsearch:
  healthcheck:
    test:
      - CMD-SHELL
      - "curl -fsS 'http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s' || exit 1"
    interval: 10s    # how often to check
    timeout: 5s      # how long to wait per check
    retries: 18      # how many failures before marking unhealthy
    start_period: 30s  # grace period at startup, failures don't count
```

And then other services declare:

```yaml
api:
  depends_on:
    elasticsearch:
      condition: service_healthy   # ← wait until ES is HEALTHY, not just STARTED
```

This is the difference between "the elasticsearch process is alive"
and "elasticsearch will respond if I query it." For real-world
systems, you almost always want `service_healthy`.

### 4.6 Environment variables: `environment` vs `env_file`

Two ways to set environment variables on a service.

```yaml
api:
  environment:                     # In version control. Safe.
    OMP_NUM_THREADS: "8"
    ELASTICSEARCH_URL: http://elasticsearch:9200
  env_file:
    - .env                         # Read from .env file. Has secrets.
```

`environment:` is for non-secret config. It lives in the compose
file in git.

`env_file:` reads a file from disk. Our `.env` (in `.gitignore`)
contains:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Compose reads it and adds those to the container's env.

**Rule of thumb**: secrets in `env_file`, config in `environment:`.

### 4.7 Volumes in compose

Recall from Chapter 2: **bind mount** maps a host directory,
**named volume** is Docker-managed.

```yaml
api:
  volumes:
    - ./data:/app/data:ro          # bind mount, read-only (:ro)
    - ./models:/app/models:ro

elasticsearch:
  volumes:
    - es_data:/usr/share/elasticsearch/data   # named volume (no leading ./)

volumes:                # named volumes must be declared at the top
  es_data:
```

Why `:ro` (read-only) on `data/` and `models/`? Defense in depth.
The API never *writes* to those directories. Marking them read-only
means a compromised process can't tamper with our data.

### 4.8 Putting it all together

Skeleton of our `docker-compose.yml`:

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS 'http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s' || exit 1"]
      interval: 10s
      retries: 18
      start_period: 30s

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      ELASTICSEARCH_URL: http://elasticsearch:9200
      OMP_NUM_THREADS: "8"
    env_file: [.env]
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
    depends_on:
      elasticsearch:
        condition: service_healthy
      redis:
        condition: service_healthy

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    environment:
      API_URL: http://api:8000
    ports: ["8501:8501"]
    depends_on:
      api:
        condition: service_started

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: trialmind
    ports: ["3000:3000"]
    volumes:
      - ./infrastructure/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./infrastructure/grafana/dashboards:/var/lib/grafana/dashboards:ro

volumes:
  es_data:
```

The dependency chain — `api` waits for ES + Redis; `ui` waits for
api — ensures startup order is correct without any user-side
scripting.

### Recap

- Compose runs many services with one command.
- Inside the network, use service names. Outside, use `localhost`
  + published port.
- Use `service_healthy`, not `service_started`, when you need a
  service to actually be ready.
- Use `environment:` for config, `env_file:` for secrets.
- Use `:ro` bind mounts for things the container shouldn't write.

### Check yourself

1. From the `api` container, what address reaches Elasticsearch?
   What about from your laptop's browser?
2. Why use `service_healthy` rather than the default `service_started`?
3. If you `git clone` the repo on a new machine, what file do you
   need to create before `docker compose up` will work?

---

## Chapter 5 — The Streamlit UI

> **By the end of this chapter, you'll be able to:**
> 1. Explain Streamlit's "rerun on every interaction" model.
> 2. Use `session_state` to persist values across reruns.
> 3. Read our UI code and identify the key patterns.

### 5.1 The Streamlit programming model

Streamlit is unusual. Every time the user interacts with a widget
(clicks a button, types in a text box), **the entire script runs
again from top to bottom**.

```python
# app.py
import streamlit as st

count = 0  # this resets to 0 on every rerun!

if st.button("Click me"):
    count += 1   # this only runs the rerun where the button was clicked
    
st.write(f"Count: {count}")
```

If you run this and click the button, **count is always 1**, never
2 or 3. Why? Because `count = 0` runs at the top *every time*. The
script has no memory.

To remember things across reruns, use `session_state`:

```python
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Click me"):
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")
```

Now `count` survives between reruns. This is a fundamental pattern
in Streamlit.

### 5.2 The example button trick

In our UI, clicking an example chip should:
1. Fill the search box with the example text.
2. Trigger a search automatically.

But we just learned that the script reruns top-to-bottom. The text
input is rendered before the button. How do we make the text input
show the example text when the button is clicked?

**Trick**: use `on_click` callbacks. They run *before* the rerun:

```python
def fill_example(text):
    st.session_state.search_input = text
    st.session_state.pending_query = text

st.text_input("Search", key="search_input")  # reads from session_state.search_input

st.button(
    "Stage 3 lung cancer, tried chemo",
    on_click=fill_example,
    args=("Stage 3 lung cancer, tried chemo",),
)

# Later in the script:
if st.session_state.pending_query:
    do_the_search(st.session_state.pending_query)
    st.session_state.pending_query = None
```

The flow:

```
User clicks button
   ↓
on_click fires (fill_example runs)
   ↓
session_state.search_input is now "Stage 3 lung cancer..."
session_state.pending_query is now "Stage 3 lung cancer..."
   ↓
Streamlit reruns the script top-to-bottom
   ↓
text_input renders, reading session_state.search_input → shows the example
   ↓
"if pending_query" branch fires → search runs
   ↓
session_state.pending_query cleared
```

This is the heart of how the example chips work in our UI.

### 5.3 Caching expensive things across reruns

Recall: the script reruns top-to-bottom on every interaction. If
we create a new HTTP client every time, we'd be opening hundreds
of connections per minute.

```python
@st.cache_resource(show_spinner=False)
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=60)
```

`@st.cache_resource` says: "create this once per session, reuse the
same object across all reruns." Now we have a single, persistent
HTTP client.

### 5.4 Showing the user what the AI understood

A nice UX touch: when the agent extracts structured fields from a
natural-language query, we show them back. Builds trust.

```python
profile = data.get("patient_profile") or {}
if any(profile.get(k) for k in ("condition", "age", "biomarkers", ...)):
    with st.container(border=True):
        st.markdown("**🧠 Here's what I understood:**")
        if profile.get("condition"):
            st.markdown(f"**Condition:** {profile['condition']}")
        if profile.get("biomarkers"):
            st.markdown(f"**Biomarkers:** {', '.join(profile['biomarkers'])}")
        # ...
```

If the user typed "HER2+ breast cancer immunotherapy", they see:

> 🧠 Here's what I understood:
> **Condition:** breast cancer  ·  **Biomarkers:** HER2-positive

If we got something wrong, they can see immediately and rephrase.

### 5.5 The match score: rank-based

Ranking systems use different score units (BM25, RRF, LightGBM
output). They're not comparable across pipelines. So instead of
showing a "real" score, we show **rank-based percentage**:

```python
def _match_score(idx: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - 0.5 * (idx / max(1, total - 1))
```

- Top result: 100%
- Last in top-K: 50%
- Linear in between

Trial #1 shows "Match 100%". Trial #5 (out of 10) shows "Match 78%".
Trial #10 shows "Match 50%". Apples-to-apples regardless of which
search backend produced the ranking.

### Recap

- Streamlit reruns the entire script on every interaction.
- Use `session_state` to remember things across reruns.
- Use `on_click` callbacks when a button needs to update widgets
  *before* the next render.
- Use `@st.cache_resource` for expensive objects (HTTP clients,
  ML models).
- Showing the user what the AI extracted builds trust.

### Check yourself

1. Why does setting a variable at the top of a Streamlit script not
   "remember" it across button clicks?
2. What's the difference between `@st.cache_data` and
   `@st.cache_resource`? (Hint: it's about whether the value is
   serializable.)
3. Why do we use a rank-based match percentage instead of the raw
   ML score?

---

## Chapter 6 — Observability

> **By the end of this chapter, you'll be able to:**
> 1. Explain what *metrics*, *scraping*, and *time-series* mean.
> 2. Describe the difference between a Counter and a Histogram.
> 3. Explain why high-cardinality labels are dangerous.
> 4. Read our `monitoring.py` and understand what it does.

### 6.1 Why observability?

Imagine your demo is live. Someone reports "the search is slow."

Without observability, you have one tool: print statements you
add and redeploy. Slow.

**With observability**, you open a Grafana dashboard:
- The latency p95 panel jumped from 2s to 12s at 14:32.
- The "search-stage latency by stage" panel shows `parse_query`
  jumped from 1.5s to 11s at 14:32.
- The error-rate panel is flat at 0%.

Conclusion: Anthropic's API got slow. Not our problem. Tell the
user.

That whole investigation took 15 seconds. That's the value of
observability: **answering "what is happening right now?" without
modifying code.**

### 6.2 The pull model

There are two ways to collect metrics:

- **Push**: your application sends metrics to a collector.
- **Pull**: the collector polls your application for metrics.

Prometheus is **pull**. Every 10–30 seconds, Prometheus does:

```
GET http://api:8000/metrics
```

Your application returns a text response describing all current
metric values. Prometheus parses it and stores the values.

Each poll is called a **scrape**.

### 6.3 What does `/metrics` look like?

```
# TYPE trialmine_requests_total counter
trialmine_requests_total{method="GET",endpoint="/health",status="200"} 142.0
trialmine_requests_total{method="POST",endpoint="/api/v1/search",status="200"} 18.0

# TYPE trialmine_request_duration_seconds histogram
trialmine_request_duration_seconds_bucket{method="POST",endpoint="/api/v1/search",le="0.1"} 0
trialmine_request_duration_seconds_bucket{method="POST",endpoint="/api/v1/search",le="1.0"} 5
trialmine_request_duration_seconds_bucket{method="POST",endpoint="/api/v1/search",le="10.0"} 18
trialmine_request_duration_seconds_count{method="POST",endpoint="/api/v1/search"} 18
trialmine_request_duration_seconds_sum{method="POST",endpoint="/api/v1/search"} 87.4
```

Some things to notice:

1. Each metric has a **name** (`trialmine_requests_total`).
2. Each measurement has **labels** in `{...}` (`method`, `endpoint`, etc.).
3. Each unique combination of labels creates a separate **time-series**.

### 6.4 Counter vs Histogram

Two types of metric you'll use most.

**Counter** — a number that only goes up.

```python
REQUEST_COUNT = Counter(
    "trialmine_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

# In a middleware:
REQUEST_COUNT.labels(method="POST", endpoint="/search", status="200").inc()
```

When you call `.inc()`, the counter increments. To get a **rate**
(requests per second), you ask Prometheus:

```
rate(trialmine_requests_total[1m])
```

That gives you "events per second over the last minute."

**Histogram** — buckets observed values.

```python
REQUEST_DURATION = Histogram(
    "trialmine_request_duration_seconds",
    "Request duration",
    ["method", "endpoint"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# In middleware, after request finishes:
REQUEST_DURATION.labels(method="POST", endpoint="/search").observe(2.3)
```

When you call `.observe(2.3)`, every bucket whose upper bound is
**≥ 2.3** ticks up. So buckets `2.5`, `5.0`, `10.0`, `30.0` all
increment.

To get a **percentile** (the latency below which 95% of requests
fall), you ask:

```
histogram_quantile(0.95, sum by (le) (rate(trialmine_request_duration_seconds_bucket[5m])))
```

This is how the "p95 latency" panel works.

### 6.5 The cardinality trap

Each unique combination of labels creates a new time-series.

Suppose we labelled `endpoint` with the *raw* URL path:

```python
endpoint=/api/v1/trial/NCT00012345    ← time-series #1
endpoint=/api/v1/trial/NCT00012346    ← time-series #2
endpoint=/api/v1/trial/NCT00012347    ← time-series #3
...
endpoint=/api/v1/trial/NCT06542055    ← time-series #140,723
```

140,723 time-series for one endpoint. Prometheus dies. Grafana crawls.

**Fix**: normalize the path to a route template.

```python
_NORMALISE_PREFIXES = (
    ("/api/v1/trial/", "/api/v1/trial/{nct_id}"),
)

def _endpoint_label(path: str) -> str:
    for prefix, template in _NORMALISE_PREFIXES:
        if path.startswith(prefix):
            return template
    return path
```

Now there's exactly **one** time-series for trial-detail requests:
`endpoint=/api/v1/trial/{nct_id}`.

> **Rule**: never use a label whose value comes from user-controlled
> input or from a high-cardinality field. Always normalize paths to
> route templates.

### 6.6 Three pieces of `monitoring.py`

The file has three responsibilities:

```
┌─────────────────────────────────────────────────┐
│              monitoring.py                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Define the metrics                          │
│     REQUEST_COUNT (Counter)                     │
│     REQUEST_DURATION (Histogram)                │
│     STAGE_DURATION (Histogram)                  │
│                                                  │
│  2. metrics_middleware(request, call_next)      │
│     For every request, before+after handler:    │
│       - record duration                         │
│       - record count by status                  │
│       - skip /metrics (avoid feedback loop)     │
│                                                  │
│  3. record_agent_trace(trace)                   │
│     For each step in the agent trace:           │
│       STAGE_DURATION.labels(stage=...)          │
│                      .observe(duration)         │
│                                                  │
└─────────────────────────────────────────────────┘
```

These are wired into FastAPI in `app.py`:

```python
from prometheus_client import make_asgi_app
from TrialMine.monitoring import metrics_middleware

app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware)
app.mount("/metrics", make_asgi_app())
```

### 6.7 Grafana provisioning — auto-loading the dashboard

When you start a fresh Grafana, it boots empty. The user has to:
1. Add Prometheus as a datasource (manual).
2. Import the dashboard JSON (manual).

For a one-command demo, that's friction. **Provisioning** lets us
auto-load both.

Two YAML files in `infrastructure/grafana/provisioning/`:

```yaml
# datasources/datasource.yml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
```

```yaml
# dashboards/dashboards.yml
apiVersion: 1
providers:
  - name: TrialMine
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

Both files mount into the Grafana container. Grafana reads them on
startup, configures itself, loads the dashboard JSON. Zero clicks.

### Recap

- Metrics are `(name, labels, value, timestamp)` tuples.
- Prometheus **pulls** by scraping `/metrics` periodically.
- **Counter** counts events; you query rates with `rate(...)`.
- **Histogram** buckets values; you query percentiles with
  `histogram_quantile(...)`.
- Each unique label combo = a new time-series. Normalize high-cardinality
  fields (NCT IDs, request IDs) to a fixed template.
- Grafana provisioning auto-loads datasources and dashboards.

### Check yourself

1. If you label requests with `user_id`, what happens after 10,000
   users?
2. Why doesn't a Counter let you query latency percentiles directly?
3. Why does the metrics middleware skip `/metrics` itself?

---

## Chapter 7 — Cold Starts and Threading

> **By the end of this chapter, you'll be able to:**
> 1. Define *cold start* and explain why it matters.
> 2. Recognize a threading bottleneck for ML inference.
> 3. Pre-warm a service to avoid cold starts.

### 7.1 What is a cold start?

The first time you run a Python program, it loads imports, JIT-compiles
PyTorch ops, opens database connections, downloads model weights.
This costs time. Subsequent runs reuse what's already loaded — fast.

The first call: **cold**.
Subsequent calls: **warm**.

For our agent, the first request paid:
- ~5s loading the cross-encoder model from disk.
- ~7s on the first cross-encoder inference (PyTorch JIT compilation).

= 12s of cold-start tax on the first user request. Painful.

### 7.2 Threading: how PyTorch uses CPU cores

PyTorch's matrix multiplications can run in parallel across CPU
cores using **OpenMP**. The number of threads is controlled by
the environment variable `OMP_NUM_THREADS`.

```
OMP_NUM_THREADS=1  → single-threaded inference, ~30 s for 50 candidates
OMP_NUM_THREADS=8  → parallel inference,        ~4 s for 50 candidates
```

Our project's CLAUDE.md set `OMP_NUM_THREADS=1` because of a macOS
issue: when faiss-cpu and Apple's Accelerate framework both load
their own copy of OpenMP, they collide and crash. Setting `=1`
neutralizes parallelism in both, dodging the crash.

But **inside a Linux container**, that crash doesn't happen. Both
libraries link against the same `libgomp1` and play nicely. So
`=1` gratuitously slows things down.

In `docker-compose.yml`:

```yaml
api:
  environment:
    OMP_NUM_THREADS: "8"   # safe in Linux container
```

### 7.3 Pre-warming: moving cold start out of the user's request

The cold-start tax is paid *somewhere*. The question is **when**:

**Option A**: pay it on the first user request. (User waits 14s.)

**Option B**: pay it during startup. (Startup takes 14s longer; users
all see warm requests.)

Option B is better for any system with real users. The way to do
this in FastAPI: add code to the **lifespan** that runs the
expensive operation once.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... load ES, FAISS, embedder ...

    # Pre-warm: run a single throwaway search to load CE + LightGBM
    if app.state.pipeline is not None:
        try:
            from TrialMine.agents.tools import (
                _get_blender_or_none,
                _get_hybrid,
                _get_reranker_or_none,
            )

            hybrid = _get_hybrid()
            reranker = _get_reranker_or_none()
            blender = _get_blender_or_none()

            if reranker is not None and hybrid is not None:
                hybrid.full_pipeline(
                    query="warmup",
                    reranker=reranker,
                    blender=blender,
                    top_k=1,
                    rerank_top_k=5,
                )
                logger.info("Pre-warmed CE + blender")
        except Exception as exc:
            logger.warning("Pre-warm failed (non-fatal): %s", exc)

    yield   # ← FastAPI starts handling requests AFTER yield
    # cleanup goes after yield
```

Adding ~3 seconds to startup. First user request is warm.

This works because the ML models are stored as **module-level
singletons** — once `_get_reranker_or_none()` loads the model, the
result is cached in a module variable. Any subsequent call returns
the same object.

### 7.4 The wrong-singleton bug (worth knowing)

When we first wrote pre-warm, we used:

```python
app.state.hybrid_retriever.full_pipeline(...)   # ← WRONG
```

It didn't help. The first request was still slow.

The reason: there are **two retrievers** in our system:

```
┌───────────────────────┐         ┌────────────────────┐
│  app.state.hybrid_    │         │  tools._get_hybrid │
│  retriever            │         │  ()                │
│                       │         │                    │
│  Built in lifespan    │         │  Built lazily on   │
│  Used by: legacy      │         │  first agent call  │
│  hybrid path          │         │  Used by: agent    │
│  (use_agent=False)    │         │  orchestrator      │
└───────────────────────┘         └────────────────────┘
```

They share the cross-encoder and LightGBM singletons (which are
also module-level), but they have **different embedder caches**.

Pre-warming `app.state.hybrid_retriever` warmed CE and LightGBM,
but the orchestrator's retriever still had to cold-start its own
embedder on the first agent call.

**Fix**: warm the same retriever the orchestrator will use.

```python
hybrid = _get_hybrid()                # ← the orchestrator's retriever
hybrid.full_pipeline(...)             # ← warmed
```

This is a generally important debugging lesson:

> When optimizing, trace through to the **actual call site**, not
> to a lookalike. Two objects of the same class can have completely
> independent caches.

### Recap

- Cold start = first-call latency penalty for loading heavy resources.
- `OMP_NUM_THREADS` controls PyTorch parallelism on CPU.
- Pre-warming runs a throwaway operation in lifespan to move cold
  cost out of the user's first request.
- Singletons must be addressed at the **consumer** — pre-warming
  the wrong one helps no one.

### Check yourself

1. Why does `OMP_NUM_THREADS=1` work on macOS host but slow things
   down in Linux container?
2. Where in our `app.py` does pre-warm run, and why is that location
   important?
3. If `app.state.hybrid_retriever.full_pipeline(...)` warms CE +
   LightGBM but not the orchestrator's embedder, what does that
   tell you about how `_get_reranker_or_none()` is implemented?

---

## Chapter 8 — The Bug Hunt

> **By the end of this chapter, you'll be able to:**
> 1. Diagnose a filter that doesn't seem to apply.
> 2. Identify a logging configuration issue.
> 3. Recognize the difference between bugs that are easy to find
>    and bugs that hide.

This chapter is about three bugs we hit while bringing the stack
up. They're all real. Each one teaches something different.

### 8.1 Bug 1 — The COMPLETED-trials filter leak

**Symptom**: User searches "Stage 3 lung cancer, tried chemo".
Returns 8 COMPLETED trials, 1 RECRUITING, 1 UNKNOWN.

**Expected**: Mostly RECRUITING (the agent's default filter is
`status=RECRUITING`).

**Step 1: confirm the filter was built.**

The agent_trace shows:

```
build_filters: {"status": "RECRUITING"}    ✓ filter exists
```

**Step 2: confirm ES has matching data.**

```
$ curl localhost:9200/trials/_search ... status:RECRUITING ... lung cancer
8,126 matching trials                       ✓ data is there
```

**Step 3: trace the filter through the retrieval code.**

In `src/TrialMine/retrieval/hybrid.py:full_pipeline`:

```python
bm25_results = self.bm25.search(
    query=query, filters=filters, top_k=200    # ← filter passed
)

semantic_results = self.semantic.search(
    query_embedding=query_embedding, top_k=200 # ← filter NOT passed
)
```

**There it is.** Filters go to BM25, but **not** to semantic search.

Why? FAISS (the semantic backend) has no native filtering. It
returns the top-200 nearest by vector cosine, regardless of any
metadata.

**The leak**:

```
BM25 returns 200 RECRUITING trials       (filtered)
Semantic returns 200 trials, any status  (unfiltered)
   ↓ RRF merge
Top 50 candidates: mostly RECRUITING, but ~10 COMPLETED leaked in
   via the semantic side
   ↓ Cross-encoder
CE rewards detailed text — COMPLETED trials have richer eligibility
criteria, score higher
   ↓ Top 10 result
Mostly COMPLETED ❌
```

**The fix**: post-RRF, drop semantic-only candidates whose ES
metadata doesn't match the filter.

```python
fused = reciprocal_rank_fusion(bm25_results, semantic_results)
bm25_meta = {r["nct_id"]: r for r in bm25_results}

# NEW: drop semantic-only candidates that don't match filters
if filters:
    bm25_ids = set(bm25_meta)
    filtered_fused = []
    for item in fused:
        if item["nct_id"] in bm25_ids:
            filtered_fused.append(item)   # already passed BM25 filter
        else:
            doc = self.bm25.get_trial(item["nct_id"])
            if doc and all(str(doc.get(k) or "") == str(v) for k, v in filters.items()):
                bm25_meta[item["nct_id"]] = doc
                filtered_fused.append(item)
    fused = filtered_fused
```

After this fix: same query returns 10/10 RECRUITING. ✓

**Lesson**: when a filter "doesn't work," trace it through every
retrieval code path. The filter *was* applied — to one of the two
backends. The other ignored it silently.

### 8.2 Bug 2 — The wrong-singleton pre-warm (recap)

We covered this in Chapter 7. The key lesson:

> Two retrievers can share some singletons (CE, LightGBM) but not
> others (embedder cache). Pre-warming object A doesn't necessarily
> warm object B even if they share a class.

### 8.3 Bug 3 — The disappearing logs

**Symptom**: We added `logger.info("Pre-warmed CE + blender")` to
the lifespan. Restarted the container. Looked at `docker logs
trialmine-api`. The line wasn't there.

**Step 1: confirm the code ran.**

After restart, the first request took 10s instead of 13s. So *something*
was different. The pre-warm was running.

**Step 2: check Python logging config.**

In `app.py`:

```python
def main() -> None:
    """Entry point for `trialmine-serve`."""
    import uvicorn
    logging.basicConfig(level=logging.INFO, ...)   # ← here!
    uvicorn.run("TrialMine.api.app:app", ...)
```

`logging.basicConfig` configures the root logger. Without it,
loggers like our `logger.info(...)` have no handler — the log
messages go nowhere.

**Step 3: figure out why `main` isn't running.**

The Dockerfile CMD is:

```dockerfile
CMD ["uvicorn", "TrialMine.api.app:app", ...]
```

This invokes uvicorn directly, **not via `main()`**. uvicorn imports
the app module — `logging.basicConfig` is never called.

**The fix**: configure logging at module-import time, idempotently:

```python
# At the top of app.py, BEFORE any logger.info calls
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
```

`if not logging.getLogger().handlers` makes it idempotent — if
something else (e.g., `main()` running from the CLI) configures
first, this is a no-op. Either way, logs work.

**Lesson**: when your code can be loaded by something other than
your CLI entrypoint (uvicorn, a test runner, a Jupyter notebook),
side effects in `main()` won't run. Anything required for the
module to function correctly belongs at module scope, not in
`main()`.

### Recap

- The filter bug: a bug can be real even when a config dict says
  "filter applied." Trace through every retrieval code path.
- The singleton bug: pre-warming the wrong object teaches you to
  always trace through to the **call site**, not to a lookalike.
- The logging bug: when your code runs under multiple entrypoints
  (CLI, uvicorn, tests), put critical setup at module scope, not
  in `main()`.

### Check yourself

1. If you saw "filter dict was built correctly" in the trace but
   the results don't reflect the filter, where would you start
   debugging?
2. Why is `if not logging.getLogger().handlers:` an idempotent
   pattern? What would happen without it?
3. If the orchestrator and `app.state.hybrid_retriever` had been
   the *same* object, would the wrong-singleton pre-warm bug have
   happened?

---

## Chapter 9 — Running the Stack

> **By the end of this chapter, you'll be able to:**
> 1. Bring the stack up from a fresh git clone.
> 2. Verify each service is healthy.
> 3. Diagnose common startup problems.

### 9.1 Prerequisites

Before you start:

- **Docker Desktop** installed (Apple Silicon or Intel Mac, or
  Linux). Allocate at least **4 GB of RAM** (Docker → Settings →
  Resources). Elasticsearch alone needs 1.5 GB.
- **The data files**: `data/trials.db`, `data/trial_embeddings.faiss`,
  `data/trial_embeddings.json`, plus the `models/` directory. These
  are produced by earlier weeks' scripts; if you don't have them,
  see the project README.
- **An Anthropic API key** (`sk-ant-...`).

### 9.2 One-time setup

```bash
# 1. Clone the repo
git clone <repo-url> TrialMine
cd TrialMine

# 2. Create .env with your Anthropic key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### 9.3 Bring everything up

```bash
docker compose up --build
```

What happens:

```
1. Docker downloads/uses cached images
   - elasticsearch:8.12.0       (~1.3 GB, downloaded once)
   - redis:7-alpine             (~60 MB)
   - prom/prometheus:latest     (~530 MB)
   - grafana/grafana:latest     (~1.4 GB)

2. Docker BUILDS our images
   - trialmine-api               (~4 GB, takes 5-10 min on first build)
   - trialmine-ui                (~820 MB, takes ~30 sec)

3. Compose starts services in dependency order
   - elasticsearch + redis     (start immediately)
   - api                       (waits for ES + redis healthy)
   - ui, prometheus            (wait for api started)
   - grafana                   (waits for prometheus)

4. Each service runs its lifespan / init
   - api: connects to ES, loads FAISS, pre-warms CE, starts uvicorn
   - prometheus: starts scraping api:8000/metrics
   - grafana: provisions datasource and dashboard
```

Total time on first run: ~10 minutes. Subsequent runs: ~30 seconds
(images cached).

### 9.4 Build the Elasticsearch index (one-time, after first compose up)

The `es_data` named volume starts empty. Index our 140K trials:

```bash
docker compose run --rm \
  -v "$(pwd)/scripts:/app/scripts:ro" \
  api \
  python /app/scripts/build_index.py \
    --es-url http://elasticsearch:9200 \
    --db /app/data/trials.db \
    --skip-semantic
```

What this does:
- `docker compose run` spawns a fresh API container (`--rm` removes
  it when done).
- `-v ".../scripts:/app/scripts:ro"` bind-mounts our scripts dir
  read-only into the container (since `scripts/` is in `.dockerignore`,
  it's not in the image).
- The script connects to ES via the service hostname, indexes 140K
  trials, exits.

Takes ~1 minute. The index persists in the named volume — survives
`docker compose down`.

### 9.5 Verify it works

```bash
# API health
curl http://localhost:8000/health
# → {"status":"ok"}

# UI
open http://localhost:8501

# Prometheus targets — should show api UP
open http://localhost:9090/targets

# Grafana — admin / trialmind
open http://localhost:3000
```

In the UI, click an example chip. The first request takes ~10s
(despite pre-warm, some additional warming happens on first real
search). Subsequent requests should be ~5–10s.

Hit the API a few times, then go to Grafana → Dashboards →
"TrialMine — API & Search Pipeline". The four panels should populate
after ~30 seconds.

### 9.6 Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `compose up` errors with "no such file: .env" | Missing .env file | Create one with `ANTHROPIC_API_KEY=...` |
| API healthcheck never goes green | `data/trial_embeddings.faiss` missing | Run `scripts/build_index.py` (without `--skip-semantic`) on host |
| First search returns "no such index [trials]" | Fresh ES volume, no data | Run the indexing step in section 9.4 |
| First search times out at 15 s | Cold start tax (rare with pre-warm but can happen on slow disks) | Send a second request — it'll be warm |
| Search returns lots of COMPLETED trials | `hybrid.py` doesn't have the filter fix | `docker compose build api && docker compose up -d` |
| Grafana shows "No data" | Haven't sent search requests yet | Send a few; wait 30s |
| `compose up` fails with disk-full or I/O error | Docker out of disk space | `docker system prune -a`; if Docker daemon is wedged, restart Docker Desktop |

### 9.7 Stop the stack

```bash
# Stop, keep data
docker compose down

# Stop and wipe ALL data (start over fresh)
docker compose down -v
```

### Recap

- One command brings up six services.
- `docker compose run` is for one-shot operations like indexing.
- The first run is slow (image build + ES startup); subsequent runs
  are fast.
- Volume data persists across `compose down`. Use `down -v` to wipe.

### Check yourself

1. Why does the indexing step use `docker compose run` instead of
   adding the indexer as a service in `docker-compose.yml`?
2. After `docker compose down`, is your ES index still there? After
   `docker compose down -v`?
3. The UI is on port 8501. Where does this port come from — the
   Streamlit code, the Dockerfile, or compose?

---

## Appendix A — File Map

### New files in this session

```
.
├── Dockerfile                                           — API image (multi-stage)
├── Dockerfile.ui                                        — UI image (single-stage)
├── .dockerignore                                        — build-context exclusions
├── docker-compose.yml                                   — six-service stack
├── infrastructure/
│   ├── prometheus/
│   │   └── prometheus.yml                              — scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/datasource.yml              — auto-load datasource
│       │   └── dashboards/dashboards.yml               — dashboard provider
│       └── dashboards/
│           └── trialmine.json                          — 4-panel dashboard
└── docs/
    └── 07_deployment.md                                — this file
```

### Modified files

| File | What changed |
|---|---|
| `src/TrialMine/api/app.py` | env-driven `ELASTICSEARCH_URL`; `/metrics` mount; pre-warm in lifespan; logging at module scope |
| `src/TrialMine/api/routes.py` | `record_agent_trace()` call after agent path |
| `src/TrialMine/monitoring.py` | TODO stub → real instrumentation |
| `src/TrialMine/retrieval/hybrid.py` | post-RRF filter fix |
| `src/TrialMine/ui/app.py` | full demo-grade rewrite |
| `CLAUDE.md` | Phase 7 update + Decisions 24–28 |

### Key functions

| Function | File | One-line description |
|---|---|---|
| `metrics_middleware` | monitoring.py | Records counter+histogram per request |
| `record_agent_trace` | monitoring.py | Pulls agent_trace step durations into STAGE_DURATION histogram |
| `metrics_app` | monitoring.py | Returns prometheus_client ASGI app for mounting at `/metrics` |
| `_endpoint_label` | monitoring.py | Normalizes high-cardinality paths to route templates |
| `lifespan` | api/app.py | Loads ES, FAISS, embedder, agent pipeline, pre-warms CE+blender |
| `_run_search` | ui/app.py | Posts to API, maps errors to friendly messages |
| `_render_card` | ui/app.py | Per-trial result card with badges, match%, expanders |
| `_match_score` | ui/app.py | Rank-based 1.0 → 0.5 linear |
| `full_pipeline` (patched) | retrieval/hybrid.py | Now drops semantic-only candidates that fail filter |

---

## Appendix B — Five Design Decisions, Plain English

### Decision 24 — Streamlit over React

**What**: Use Streamlit (Python) for the UI instead of React (JavaScript).

**Why**: Streamlit ships in one Python file. No build tooling, no API
client to write, no state management library. Faster for a demo.

**Tradeoff**: Streamlit doesn't reflow well on mobile, has limited
customization, and its "rerun the whole script" model fights real-time
streaming UIs.

**Migrate when**: real concurrent users, mobile-first, or
streaming-token UIs.

### Decision 25 — Docker Compose over Kubernetes

**What**: Use Compose for a single-host demo. Don't reach for K8s.

**Why**: One YAML file, one command. No cluster, no operators.

**Tradeoff**: Compose YAML doesn't translate to K8s manifests.
Migration is via Helm or Kustomize, not copy-paste.

**Migrate when**: more than one host, autoscaling, rolling deploys.

### Decision 26 — Elasticsearch heap = 1 GB

**What**: `ES_JAVA_OPTS=-Xms1g -Xmx1g`.

**Why**: Comfortable for our 140K docs (~600 MB index). 512 MB sits
below ES 8.x's recommended floor for non-trivial datasets.

**Honest disclaimer**: We didn't actually benchmark 512 MB OOM.
The choice is precaution, not measurement. If RAM is constrained,
768 MB is the next thing to try.

### Decision 27 — `OMP_NUM_THREADS=8` in Linux container

**What**: Override the project's macOS default of `=1` to `=8`
inside the API container.

**Why**: `=1` was set to avoid a macOS-specific OpenMP collision
between FAISS and Apple's Accelerate. Inside Linux containers,
that collision doesn't happen, and `=1` cripples PyTorch CE
inference (30s instead of 4–8s for 50 candidates).

**Caveat**: Keep `=1` for `python scripts/...` runs on macOS host.

### Decision 28 — Filter at post-RRF candidate level

**What**: After RRF merge, drop semantic-only candidates whose ES
metadata doesn't match the user's filter.

**Why**: FAISS has no native filter; passing filters only to BM25
let COMPLETED trials leak in via the semantic side. Post-RRF
filtering is simple and adds 50–100 extra ES gets per query
(~250–500 ms).

**Alternative considered**: pre-filter the semantic side using ES
mget. Would be slightly faster but couples the retriever to ES's
mget API. Rejected as premature optimization.

---

## Appendix C — Glossary

- **container** — running, isolated process tree on the host kernel
- **image** — frozen, layered filesystem; containers are instances
- **layer** — single content-addressable diff; cached for fast rebuilds
- **multi-stage build** — Dockerfile with multiple `FROM` stages, copy
  only the runtime essentials into the final image
- **`.dockerignore`** — patterns excluded from the build context
- **bind mount** — host directory mapped into a container
- **named volume** — Docker-managed persistent storage
- **healthcheck** — periodic command determining "healthy" state
- **service hostname** — DNS name inside compose network = service name
- **ASGI** — async-native Web server interface (FastAPI, Starlette)
- **middleware** — code that wraps every HTTP request
- **scrape** — Prometheus's HTTP poll of a `/metrics` endpoint
- **time-series** — sequence of `(time, value, labels)` samples
- **histogram** (Prometheus) — bucketed counter for percentile queries
- **p95** — the 95th-percentile latency
- **cold start** — first-call latency penalty (load + JIT)
- **pre-warm** — run a throwaway operation in lifespan to skip cold
  start in the user's request
- **OpenMP / libgomp / libomp** — parallel-computing runtime
- **JIT** — just-in-time compilation; PyTorch ops compile on first use
- **RRF** — Reciprocal Rank Fusion, our BM25 + semantic merge
- **provisioning** — auto-loading config (datasources, dashboards)
- **containerd** — the lower-level runtime under Docker

---

## Appendix D — Exercises

If you want to make sure you've got it, try these. Solutions are in
the code; check yourself.

### Beginner

1. Add a new sidebar option to the Streamlit UI: a slider for
   eligibility-check strictness (1–10). The widget doesn't need to
   *do* anything, just appear.
2. Add a healthcheck to the `prometheus` service in
   `docker-compose.yml`. Hint: Prometheus has a `/-/healthy` endpoint.
3. Add a 5th panel to the Grafana dashboard: total requests in the
   last 5 minutes (`sum(increase(trialmine_requests_total[5m]))`).

### Intermediate

4. Why is the API's healthcheck running a Python one-liner instead
   of `curl`? (Hint: look at the slim runtime image.)
5. Modify the post-RRF filter to *log* (not raise) when it drops
   more than 10 candidates per query — this would help us spot
   filter-too-aggressive cases in production.
6. Add a `restart: unless-stopped` policy to all six services. What
   does it do?

### Advanced

7. Replace the post-RRF filter (one ES `get_trial` per
   semantic-only candidate) with a single ES `mget` call. Benchmark
   the latency improvement.
8. The pre-warm step uses `top_k=1, rerank_top_k=5`. Why those
   values and not `top_k=10, rerank_top_k=50` (matching real
   queries)? Make a case for either.
9. Sketch what would change to scale this from "one host" to "one
   API replica per host, 3 hosts" with Compose. Then sketch the
   same for Kubernetes. Compare.

---

*If you read this far, congratulations. You should now be able to
explain, build, run, and debug a containerized ML system end-to-end.
That's a real skill. Use it.*
