# MD. RAYHAN HOSSAIN
**Software Engineer | Backend, Distributed Systems & AI Agents**  
mostafaizurrahman2021@gmail.com · +8801312961737 · Dhaka, Bangladesh  
[LinkedIn](https://linkedin.com/in/mdrayhan03) · [GitHub](https://github.com/mdrayhan03) · [Portfolio](https://rayhan-portfolio.lovable.app)

---

## PROFESSIONAL SUMMARY
Software Engineer with production experience designing clean, maintainable backend architectures, high-throughput distributed tools, and autonomous AI agents. Strong foundation in software craftsmanship—applying Hexagonal architecture, GoF design patterns, and test pyramids to production code. Track record of cutting API latency by ~35% on live systems, building multi-provider LLM orchestration with resilient fallback, and maintaining a 3.81 CGPA while mentoring 200+ students.

---

## WORK EXPERIENCE

### **Full Stack Software Engineer** | *Janatawifi LTD*
**Dec 2025 – Present · Dhaka, Bangladesh**  
*Python · FastAPI · Django · React · PostgreSQL · Redis · Celery · Docker · Azure · LangChain · LangGraph · Frappe*

- **TraderBro (Autonomous AI Agent):** Architected the backend orchestration using **FastAPI**, **LangChain**, and **LangGraph**, integrating GPT, Claude, Gemini, and xAI with the Twitter API and ScrapeCreators for real-time financial signal aggregation.
- **Multi-Provider Fallback & Tool Calling:** Engineered resilient multi-model routing with automated function/tool calling and fallback paths across 4 LLM providers, ensuring 100% agent uptime during single-vendor API degradations.
- **Performance & Latency Optimization:** Cut average API response times by **~35%** by offloading asset storage to **Cloudflare R2 (S3-compatible)** object storage and tuning asynchronous task execution with **Celery** workers and **Redis** caching.
- **CWTAMC (Enterprise Real Estate ERP):** Developed and deployed an enterprise operations ERP on the **Frappe Framework / ERPNext**, replacing manual spreadsheet workflows, eliminating hours of repetitive daily entry, and migrating historical data.
- **Production DevOps & Monitoring:** Containerized all microservices via **Docker Compose** on **Azure**; instrumented structured Python logging and monitored container health (CPU, memory, disk) using Portainer and custom alerting.

### **Undergraduate Teaching Assistant** | *Independent University, Bangladesh*
**Feb 2024 – Nov 2025 · Dhaka, Bangladesh**  
*Java · JavaFX · Object-Oriented Programming · Data Structures · Git/GitHub*

- Mentored **200+ undergraduate students** across tutorial sessions and hands-on lab projects on **OOP fundamentals**, SOLID design, JavaFX desktop development, and collaborative Git workflows.
- Conducted regular code reviews, debugging clinics, and prepared practical laboratory exercises on data structures and algorithms.

---

## FEATURED TECHNICAL PROJECTS

### **Plastic Kothay — Crowdsourced Geo-Waste Tracking Platform**
`Django` · `Django REST Framework` · `React` · `TypeScript` · `PostgreSQL` · `Docker` · `GitHub Actions` · `pytest`  
[GitHub Repository](https://github.com/mdrayhan03/plastickothay_dev)

- **Hexagonal Architecture:** Architected the backend strictly on **Ports and Adapters (Hexagonal Architecture)**—isolating a 100% framework-free domain core behind abstract ports, with swappable adapters for persistence, auth, and storage wired via a composition root.
- **Test Pyramid & Contract Testing:** Built a comprehensive test suite across **31 test files** (unit, API integration, and SQL contract tests asserting that runtime leaderboard queries match pure domain scoring rules to prevent schema drift).
- **CI/CD Architectural Enforcement:** Configured automated GitHub Actions workflows running `lint-imports` to strictly forbid web/ORM dependencies from leaking into the domain core, alongside `ruff` and `pytest`.
- **Compensating Rollback:** Implemented reliable external storage synchronization with automated compensating rollbacks for failed media uploads since external APIs cannot participate in database ACID transactions.

### **ShareBridge — High-Throughput P2P LAN File Sharing & Messaging**
`Python` · `AsyncIO` · `Aiohttp` · `WebSockets` · `KivyMD 2.0` · `Pytest`  
[Download & Live Demo](https://mdrayhan03.github.io/ShareBridge/) · [GitHub Repository](https://github.com/mdrayhan03/ShareBridge)

- **High-Throughput Binary Streaming:** Engineered an offline-first LAN communication platform capable of streaming **20GB+ files** without memory bloat using `aiohttp` non-blocking chunked byte streams and peer-to-peer local HTTP servers.
- **Zero-Config Discovery & Security:** Built automated local network peer discovery over **UDP broadcast beacons (port 37020)** with loopback probes; secured transfers via an ephemeral **tokenized file registry** (`/download/<token>`) that strictly prevents local filesystem path disclosure.
- **Cross-Platform Delivery:** Shipped unified client packages for **Windows, macOS, Linux, and Android (APK)** powered by automated GitHub Actions build pipelines triggered by version tags.

### **RideBuddy — Campus Ride-Sharing & Cost-Splitting Engine**
`Django` · `PostgreSQL` · `OpenStreetMap API` · `HTML5/CSS3` · `JavaScript`

- Designed custom ride-matching and fair cost-splitting algorithms with gender and route preference filters to reduce commute expenses.
- Integrated OpenStreetMap API for route tracing and managed complete ride lifecycle state transitions with comprehensive Django unit tests.

---

## APPLIED RESEARCH & THESIS

### **Undergraduate Thesis: Multi-Target Biomass Estimation with Vision Transformers**
*Independent University, Bangladesh · Computational Physics Lab*  
`Python` · `TensorFlow` · `Keras` · `Vision Transformer (ViT-B16)` · `EfficientNet` · `Grad-CAM` · `SHAP` · `LIME`

- Benchmarked 5 deep learning backbones (**ViT-B16, EfficientNet-B0/B3, DenseNet-121, MobileNet**) under a unified transfer learning pipeline for multi-target regression from single RGB images.
- Designed a custom regression head featuring **CBAM attention** and composite MAE/RMSE loss, with **ViT-B16 achieving the highest weighted R² (0.746)**; validated statistical significance using bootstrap confidence intervals and explainable AI (Grad-CAM, SHAP).

---

## TECHNICAL SKILLS

- **Languages:** Python (Advanced), SQL (PostgreSQL, MySQL, MariaDB), JavaScript, TypeScript, Java, Bash
- **Backend & APIs:** Django, Django REST Framework (DRF), FastAPI, Flask, Celery, Redis, RESTful API Design, JWT, OAuth 2.0, Frappe / ERPNext, WebSockets
- **Software Architecture & Design:** Hexagonal (Ports & Adapters) Architecture, Clean Architecture, SOLID Principles, GoF Design Patterns, Domain-Driven Design (DDD) concepts, Event-Driven Task Processing
- **AI & Agentic Systems:** LangChain, LangGraph, Multi-Agent Workflows, LLM Tool Calling & Routing, RAG (Retrieval-Augmented Generation), Qdrant (Vector DB), Prompt Engineering
- **Databases & Storage:** PostgreSQL, MySQL, MariaDB, MongoDB, Redis, Cloudflare R2 / S3 Object Storage, Alembic Migrations
- **DevOps & Infrastructure:** Docker, Docker Compose, Linux Administration, Azure, Nginx Reverse Proxy & Load Balancing, GitHub Actions (CI/CD), Portainer
- **Testing & Quality Assurance:** pytest, unittest, Contract Testing, Integration Testing, Playwright E2E, Vitest

---

## EDUCATION

**Bachelor of Science in Computer Science & Engineering**  
*Independent University, Bangladesh (IUB)* · **Sep 2022 – Expected Sep 2026**  
- **CGPA: 3.81 / 4.00** · **Dean's List Honoree**  
- Key Coursework: Data Structures & Algorithms, Object-Oriented Programming, Database Systems, Operating Systems, Software Engineering, Discrete Mathematics.

---

## HONORS & CERTIFICATIONS

- **1st Runner-Up** — Hult Prize IUB (Feb 2025)
- **1st Runner-Up** — Cisco IoT Hackathon (Nov 2024)
- **2nd Runner-Up** — IUB Intra Programming Contest (Nov 2023)
- **Dean's List Honoree** — Independent University, Bangladesh (Dec 2022)
- **Certifications:** Docker Essentials (IBM), CCNA: Introduction to Networks (Cisco), Agile Project Management (HP LIFE)

