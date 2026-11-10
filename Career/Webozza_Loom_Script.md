# 🎥 Webozza Loom Script — walkthrough of plastickothay_dev

> **Goal:** Show *how you think*, not a polished performance. Every section is written to demonstrate a **decision + reasoning** — that's what beats their "we don't want someone who just prompts AI" filter.
> **Format:** Two parts, ~5 min each (free Loom limit). Screen-share the actual code as you talk.
> **Golden rule:** Whenever you name a technology, immediately say *why you chose it over the alternative*. Reasoning > features.
>
> **Delivery tips:**
> - Don't script word-for-word — use these as talking points so it sounds natural.
> - Have the repo open in your editor; navigate to the file you're describing as you speak.
> - It's fine to pause, think, and say "the reason I did this was…". That *is* the signal they want.
> - Slightly rough and thoughtful beats slick and shallow.

---

## PART 1 — Intro, Problem, Architecture, Database (~5 min)

### 0:00–0:40 · Intro + production anchor
**Say:**
> "Hi, I'm Rayhan, a full-stack engineer working mainly in Python and React. Professionally I build production systems — an autonomous multi-LLM AI agent and an enterprise ERP — but those are private client codebases. So I'm going to walk you through a project that's fully mine and public, because it lets me show you every architectural decision in the actual code. It's a crowdsourced plastic-waste reporting platform. I built it to solve a real problem *and* to practice production-grade engineering."

**Why this works:** the one-line production anchor proves you ship real client work; then you give a *smart reason* for showing a personal project (you can show the code). Never call it a "hobby."

### 0:40–1:30 · What it does + the problem
**Say:**
> "The problem: plastic waste is everywhere, but there's no easy way for ordinary people to report it or for anyone to see the hotspots. So the app lets anyone submit a geo-tagged report with a photo and a severity. Moderators review submissions, approved reports show on a public map, and there's a scoring and leaderboard system that gamifies participation — you earn points when your reports get approved and when people engage with them."

**Show:** the running app / a screenshot of the map + a report, or the README.

### 1:30–2:00 · What YOU personally built
**Say:**
> "I built the whole thing end to end — the Django backend, the React/TypeScript frontend, the database design, the deployment, and the test suite. No template — the architecture is a deliberate design."

### 2:00–4:00 · Architecture + application flow (the centerpiece)
**Show:** the `backend/` folder tree — `core/domain`, `core/ports`, `adapters/`, `config/container.py`, `api/`.

**Say (walk it slowly, this is the money section):**
> "I structured the backend as a **hexagonal architecture** — ports and adapters. Let me show you why.
>
> The `core/domain` folder is pure Python — entities, value objects, the scoring rules — with **zero Django imports**. This is the business logic, and it doesn't know or care that Django exists.
>
> `core/ports` defines interfaces — abstract base classes — for everything the domain needs from the outside world: repositories, storage, notifications, security, a clock. The domain depends on these *interfaces*, not on concrete tech.
>
> `adapters/` holds the real implementations — Django ORM for persistence, Google Drive for image storage, Mailjet for email, JWT for auth. These plug into the ports.
>
> And `config/container.py` is the composition root — the one place I wire ports to concrete adapters.
>
> **Why I did this:** it means my business logic is testable without a database, and I can swap infrastructure without touching the core. For example, image storage is a one-line swap between Google Drive in production and the local filesystem in development, because both implement the same port — nothing above it changes.
>
> **One deliberate call:** I used plain factory functions for the composition root instead of a DI framework. For a monolith this size, a framework would be over-engineering — I wanted the wiring to be obvious, not hidden behind magic."

**Defense notes (if asked):**
- *"Isn't this overkill?"* → "For this app's current size, arguably yes — I'd right-size it in a smaller project. I built it this way deliberately to practice the pattern and keep the domain independently testable. Knowing *when* to apply this is as important as knowing *how*."
- *"Why hexagonal over layered?"* → "Layered still lets the domain depend on the framework. Hexagonal inverts that — the domain is the center and everything else is a plugin. It made the contract tests and swappable storage possible."

### 4:00–5:00 · Database design
**Show:** `adapters/persistence/django_orm/` — `models.py`, `mappers.py`, `repositories.py`, `unit_of_work.py`, migrations.

**Say:**
> "The database is PostgreSQL, accessed through the Django ORM — but only inside the persistence adapter. The ORM models never leak into the domain; I have **mappers** that convert between ORM models and domain entities at the boundary. The reason: if a lazy queryset crossed into the domain, it would drag persistence behavior in and make the whole port abstraction a fiction.
>
> I also wrapped writes in a **Unit of Work**, so a use case either commits everything or nothing.
>
> The most interesting database decision was the scoring system — I'll come back to that in Part 2 as a technical challenge, because it's the part I'm most proud of."

**[END PART 1 — stop recording, start Part 2]**

---

## PART 2 — APIs, Deployment, Challenges, Reflection (~5 min)

### 0:00–1:30 · APIs + third-party integrations
**Show:** `api/` folder (organized by context: auth, users, content, engagement, reports, scoring, admin), plus `adapters/storage/gdrive.py`, `adapters/notifications/mailjet.py`, `adapters/security/jwt_service.py`.

**Say:**
> "The API is Django REST Framework, organized by **bounded context** — auth, reports, engagement, scoring, admin — each with its own serializers, views, and URLs. I have a central exception handler so every endpoint returns errors in a consistent shape, plus shared pagination and permission classes.
>
> Third-party integrations, all behind ports:
> - **Google Drive** for image storage in production, local filesystem in dev.
> - **Mailjet** for transactional email/notifications.
> - **JWT** (SimpleJWT) for stateless authentication.
>
> Because each one sits behind a port, if I ever swap Mailjet for another provider, I change one adapter and nothing else."

### 1:30–2:30 · Deployment, Docker, infrastructure
**Show:** `Dockerfile`, `docker-compose`, `config/settings/` (base/dev/prod/test), `.github/workflows/backend-ci.yml`.

**Say:**
> "It's containerized with Docker, and settings are split by environment — base, dev, prod, test — so config never leaks across stages and secrets stay in the environment, never in code.
>
> The part I'm proud of on the ops side is CI. My GitHub Actions pipeline runs three gates on every push: **ruff** for linting, **pytest** for the full test suite, and — this is the unusual one — **`lint-imports`**, which *enforces the architecture*. If any code in `core` ever imports Django, the build fails. So the architecture can't quietly rot over time — it's protected by CI."

**Defense note:** this is a strong "I think about maintainability" signal — lean into it.

### 2:30–4:00 · Technical challenges + how you solved them (your best material)
**Show:** `core/domain/points.py` and `tests/contract/`.

**Say (the scoring/drift story — this is your strongest 90 seconds):**
> "The hardest problem was the scoring system. Scores had to stay correct even though moderators can hide or un-hide posts, and users can add or remove likes at any time.
>
> The obvious approach is a points ledger — a score column you increment and decrement. But I realized that leads to fragile reversal logic: hiding a post means finding and reversing its points *and* all its likes' points; un-hiding means restoring them. Every edge case needs reversal code, and that's exactly where score-corruption bugs live.
>
> So instead of *storing* the score, I **derived** it from current state — a user's score is a function of their current post statuses, current engagements, and the active rules. Hiding a post automatically strips its points because the query just stops counting it. There's no reversal code anywhere — the whole bug class disappears by design.
>
> That created a second problem: deriving on every read is expensive. So in production the leaderboard is optimized raw SQL, while the pure-Python rules stay the canonical spec. Now the same logic lives in two places and could drift apart — which is dangerous. My fix was a **contract test suite** that runs identical scenarios against both the Python rules and the SQL, and a new implementation only ships if they produce identical results. So I got SQL's performance with a single source of truth."

**(Optional second challenge if time — the upload compensation):**
> "A smaller one: image uploads to Google Drive aren't part of the database transaction. So if the DB insert failed after a successful upload, I'd orphan a file in Drive forever. I handle it with a compensation step — if the transaction fails, I delete the uploaded file, then re-raise the error."

### 4:00–4:45 · What you'd do differently
**Say (be honest — this answer is a senior signal):**
> "Two things. First, honestly, for an app this size the full hexagonal architecture is arguably more than the domain needs — I'd right-size the complexity to the problem rather than applying the pattern everywhere. Second, I'd add observability earlier — structured logging and error tracking from day one, rather than relying on basic logs. The architecture I'd keep; the *amount* of it I'd tune to the problem."

**Why this works:** self-aware "what I'd change" answers are the #1 senior signal. Admitting the over-engineering *before they point it out* shows exactly the judgment they're testing for.

### 4:45–5:00 · Close + the AI question (pre-empt it)
**Say:**
> "Last thing — I build with AI tools like Claude Code, and I think that's a strength when you drive it. But the decisions here — deriving the score instead of storing it, the contract tests, keeping the domain framework-free — those are *my* engineering calls, and I can defend every one of them. I use AI to move faster; I don't let it think for me. Thanks for watching — happy to go deeper on any part."

**Why:** this directly answers Webozza's core concern *before they ask*, and turns your AI-first workflow from a liability into a "I direct it, I own the output" strength.

---

## 🛡️ Likely follow-up questions — know these cold

| They ask | Your answer |
|---|---|
| "Why not just cache the derived score?" | "I could — but a cache adds its own invalidation problem on every hide/like. Deriving on read keeps a single source of truth; the SQL implementation already makes it fast enough. Caching would be the next step if reads became a bottleneck." |
| "How do the contract tests actually work?" | "They define scenarios — e.g. a post approved then hidden — and assert the same final score from both the Python rules and the SQL query. Any new leaderboard implementation must pass the identical suite." |
| "What happens when the point rules change?" | "Because score is derived, changing a rule just changes future calculations — there's no stored history to migrate or reverse. That's a direct benefit of not having a ledger." |
| "Why no DI framework?" | "For a monolith this size it's unnecessary indirection. Plain factory functions make the wiring explicit and readable. I'd reconsider only if the graph got large." |
| "Did AI write this?" | "I used AI to accelerate implementation, but the architecture and the hard decisions are mine — and I can walk you through the reasoning for any of them, which is what matters." |

---

## ✅ Pre-record checklist
- [ ] **Clean the WebRTC repo** (leaked creds) — they may check your GitHub.
- [ ] Have `plastickothay_dev` open in your editor, files ready to navigate.
- [ ] Do ONE practice run out loud — time each part (~5 min).
- [ ] Test mic + screen-share in Loom first.
- [ ] Record Part 1, then Part 2. Don't over-edit — thoughtful-and-real wins.
- [ ] Reply to their email with both Loom links.
