# 🎯 SahajMobile — Technical Interview Prep

> **Role:** Full-Stack SWE, backend-heavy (80% Python/FastAPI, 20% React). Product: Identity Verification (IDV) — OCR on ID cards + selfie/face matching + liveness detection.
> **Their process:** 1) Technical screening (background + stack) → 2) Deep-dive w/ CTO (system design + coding) → 3) CEO culture fit → 4) paid trial.
> **Your situation:** Strong on FastAPI/architecture/async; gaps in the IDV/vision domain (OCR, face-matching, liveness), MySQL/SQLAlchemy specifics, and years. You got the callback anyway — so lead with strengths, be honest about gaps, show you learn fast.
>
> **Golden rule for this interview:** Never fake the vision/OCR domain. They have a CTO deep-dive + a paid trial — bluffing gets exposed. Instead: "I haven't built OCR/liveness in production, but here's how I'd reason about it, and I have image-processing foundations." Honest + can-reason beats fake-confident.

---

## PRIORITY 1 — Your own projects (the screening is mostly THIS)

The first round is "a conversation about your background and stack." Be able to walk through your projects fluently, with *decisions and reasoning*.

**plastickothay_dev — your strongest card (and directly relevant):**
- They explicitly want **"Dependency Injection"** — your hexagonal/ports-and-adapters design *is* DI done properly. Lead with this.
- Walk it: framework-free domain core → ports (interfaces) → adapters (swappable implementations) → composition root wires them.
- The scoring/derived-state problem (your "hardest problem" story).
- CI that enforces architecture (lint-imports), full test pyramid.

**TraderBro — your FastAPI + async proof:**
- FastAPI backend, async services, multi-LLM agent, Celery/Redis (cut API time ~35%).
- Relevant because their APIs are FastAPI + high-concurrency + async.

**Practice:** say each project's *what → why → hardest problem → what you'd change* out loud in ~2 min.

---

## PRIORITY 2 — FastAPI + Python core (their 80% — know this cold)

They want **"Deep understanding of FastAPI, AsyncIO, Pydantic, and Dependency Injection."**

**FastAPI:**
- Routing, path/query/body params, request/response models
- **Pydantic** models, validation, `BaseModel`, field validators, settings management
- **Dependency Injection** via `Depends()` — how it works, why it's useful (DB sessions, auth, shared logic). Connect it to your hexagonal DI mindset.
- Async endpoints (`async def`), background tasks, middleware
- Auto-generated **OpenAPI/Swagger** docs
- Auth: OAuth2 password flow, JWT, dependencies for auth

**AsyncIO (study this — it's a known soft spot for you):**
- Event loop, coroutines, `async`/`await`
- `asyncio.gather`, concurrent tasks
- **When async helps** (I/O-bound: DB, HTTP, file) **vs when it doesn't** (CPU-bound — like OCR/face-matching! those should go to workers/threads, not block the event loop)
- Sync vs async DB drivers

**Python internals (your interview feedback flagged these — brush up):**
- **GIL** — what it is, why threading doesn't parallelize CPU-bound work, why async is single-threaded concurrency
- Why CPU-heavy tasks (image processing) need multiprocessing/workers, not asyncio
- Mutability, hashability, decorators, generators, context managers

---

## PRIORITY 3 — Databases: MySQL + SQLAlchemy (a real gap to close)

You're strong on **Django ORM + PostgreSQL**, but they use **MySQL + SQLAlchemy**. Close the specifics:
- **SQLAlchemy** (Core vs ORM): sessions, models, relationships, queries. Map it from your Django ORM knowledge — same concepts, different API.
- SQL fundamentals: joins, indexing, `EXPLAIN`, query optimization
- **N+1 query problem** and how to avoid it (eager loading / `joinedload`)
- Schema design, normalization, transactions, connection pooling
- "High-performance, optimized SQL queries" — be ready to reason about indexes and slow queries
- **Honest line:** "My production ORM work is Django/PostgreSQL, but the concepts transfer directly to SQLAlchemy/MySQL and I'm comfortable picking up the specifics."

---

## PRIORITY 4 — REST API design, Auth & Security

- REST best practices, resource naming, status codes, versioning, pagination, error handling
- **OpenAPI/Swagger** documentation
- **Auth:** JWT, OAuth2, and read up on **Auth0** specifically (they use it — know what it is: hosted identity/auth provider, tokens, flows). You have JWT/OAuth2; frame Auth0 as "same concepts, hosted service."
- Securing production APIs: input validation (Pydantic), rate limiting, HTTPS, secrets management
- **Security is a flagged gap for you** — brush up OWASP basics (injection, broken auth, sensitive data exposure), especially relevant for an *identity/trust* company.

---

## PRIORITY 5 — The IDV domain (research + honest framing)

This is their product and your biggest knowledge gap. **Don't fake it — but show you researched it and can reason.** You have image-processing foundations (your thesis/Learning work) — leverage that.

**OCR (extracting text from ID cards):**
- What OCR is; libraries: **Tesseract, EasyOCR, PaddleOCR**; cloud: **AWS Textract, Google Vision**
- **Image preprocessing to improve OCR:** grayscale, noise reduction, thresholding/binarization, deskew/perspective correction, cropping to the ID region
- Trade-off: local libraries (free, private, control) vs cloud OCR (accurate, easy, costs money + sends data out) — relevant for a *privacy-sensitive* IDV product

**Face matching (selfie vs ID photo):**
- Libraries: **face_recognition, DeepFace, dlib, OpenCV**
- Concept: detect face → generate an **embedding (vector)** → compare via **distance/similarity threshold** → match or not
- (Your Qdrant/vector-search background helps you *understand* embeddings — connect it)

**Liveness detection (anti-spoofing):**
- Problem: someone holds up a photo/video instead of a live face
- **Passive** (analyze one image for texture/depth cues) vs **active** (ask user to blink/turn head)
- Know it exists and why it matters for trust/security

**Your honest framing:**
> "I haven't built an OCR/face-matching pipeline in production, but I have image-processing foundations from my thesis work, I understand the pipeline conceptually — preprocess → OCR/detect → embed → match → liveness — and I'm genuinely excited to go deep on it. My strength is the *backend engineering* that wraps these — high-concurrency async APIs, offloading the heavy CV work to background workers, and clean architecture."

---

## PRIORITY 6 — System Design (for the CTO deep-dive)

Most likely question: **"How would you design our Identity Verification system?"** Prepare this — it's their core product. Sketch:

```
Client uploads ID image + selfie
        │
   FastAPI endpoint (validates, stores to object storage — NOT in the request path for heavy work)
        │
   Enqueue job (Celery/queue) — CV work is CPU-heavy, must NOT block async event loop
        │
   Worker pipeline:  preprocess → OCR (extract NID text) → face detect+embed (ID & selfie)
                     → compare embeddings (threshold) → liveness check → result
        │
   Store result (MySQL), notify client (webhook/polling/websocket)
```

**Key points to make (show senior thinking):**
- **CPU-bound CV work goes to background workers, not the async API** (ties back to your asyncio/GIL knowledge — great connection to make).
- Object storage for images (S3/Azure Blob), not the DB.
- Idempotency, retries, handling failures in the pipeline.
- Security/privacy: NID + biometric data is highly sensitive — encryption, access control, data retention. (Frame this for a *trust* company.)
- Scaling: horizontal workers, queue backpressure, caching.

Also prep general system-design basics: load balancing, caching, DB scaling, async processing (your roadmap strengths — lean in).

---

## PRIORITY 7 — React / TypeScript (the 20%)

Lighter, but be ready:
- React **Hooks** (useState, useEffect, useContext), **Context API** for state
- TypeScript basics, typing props/components
- **Next.js** (they use it), App Router basics, rendering strategies
- Note: **they use React, not Django templates** — so talk React-first for frontend.

---

## PRIORITY 8 — DevOps / Cloud / CI/CD

- **AWS or Azure** (you have Azure) — deployment lifecycle, containers
- **CI/CD** (GitHub Actions — you have this in plastickothay, a real asset)
- Docker (you're solid here)

---

## Behavioral / Fit (screening will touch this)

- **"Why SahajMobile?"** — already prepped: intersection of backend + computer vision, digital-trust mission, the IDV pipeline challenge.
- **"Product-first mindset"** — have an example where you thought about *user value*, not just code (e.g., RideBuddy cost-splitting for real commuters; CWTAMC eliminating manual Excel work).
- **Handle the experience gap honestly:** "I'm earlier in my career than the posting's 3+ years, but here's a production-grade, well-architected codebase (plastickothay) and a fast learning curve. I'd rather show you what I can build than claim years."

---

## ⚡ If you have LIMITED time, study in THIS order

1. **Your projects** — walk-throughs fluent (plastickothay DI/architecture, TraderBro FastAPI/async). *(Biggest ROI for the screening.)*
2. **FastAPI + DI + Pydantic + async** — their 80%, and where you're already strong. Sharpen it.
3. **The IDV pipeline** (Priority 5 + 6) — research it enough to reason + design it. Shows you understand their product.
4. **AsyncIO + GIL** — and the key insight: *CPU-heavy CV work goes to workers, not the async loop.* (Connects your Python knowledge to their product — a great thing to say.)
5. **SQLAlchemy/MySQL** — map from your Django/Postgres knowledge.
6. **Auth0 + security basics** — quick read, important for an identity company.

---

## Likely questions to rehearse

| Question | Your angle |
|---|---|
| "Walk me through your most significant project." | plastickothay — architecture + DI + the scoring problem |
| "How does dependency injection work in FastAPI?" | `Depends()`, + connect to your hexagonal DI design |
| "When do you use async vs sync in Python?" | I/O-bound → async; CPU-bound (OCR/face) → workers; explain GIL |
| "How would you design our IDV/OCR/face-match system?" | The pipeline in Priority 6; heavy CV work off the request path |
| "You use PostgreSQL — how comfortable are you with MySQL/SQLAlchemy?" | Concepts transfer; honest + confident |
| "Have you worked with OCR / face recognition?" | Honest: not in production, but here's how I'd reason + my CV foundations |
| "Why should we hire you at under 3 years?" | Show plastickothay; ability + learning curve over tenure |

**Final reminder:** honesty + reasoning beats bluffing. They test with a paid trial, so authenticity is your best strategy. Lead with FastAPI/architecture/async (real strengths), reason intelligently about the vision domain, and be transparent about the gaps.
